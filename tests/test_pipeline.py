"""
Tests unitaires et de bout en bout pour Face_ID_v1.
Tous les fichiers de test sont créés dans des répertoires temporaires (tempfile)
et sont automatiquement supprimés à la fin du test pour garantir que les dossiers
du projet (dataset/, models/, results/) restent intacts et prêts pour les vrais étudiants.
"""

from pathlib import Path
import os
import pickle
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.detect_faces import FaceDetector, DEFAULT_YUNET_MODEL
from src.encode_faces import FaceEncoder, encode_dataset, DEFAULT_SFACE_MODEL
from src.recognize_faces import FaceRecognizer
from src.evaluate import evaluate_dataset


class TestFaceIDPipeline(unittest.TestCase):
    """Suite de tests automatisés pour valider l'ensemble du pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initialise les composants partagés."""
        cls.detector = FaceDetector()
        cls.encoder = FaceEncoder(detector=cls.detector)

    def test_01_models_exist(self):
        """Vérifie la présence et la taille des fichiers de modèles ONNX."""
        self.assertTrue(DEFAULT_YUNET_MODEL.exists(), "Modèle YuNet manquant !")
        self.assertTrue(DEFAULT_SFACE_MODEL.exists(), "Modèle SFace manquant !")
        self.assertGreater(DEFAULT_YUNET_MODEL.stat().st_size, 50_000, "Modèle YuNet corrompu ou incomplet")
        self.assertGreater(DEFAULT_SFACE_MODEL.stat().st_size, 10_000_000, "Modèle SFace corrompu ou incomplet")

    def test_02_face_detector_on_blank_and_noisy_image(self):
        """Vérifie le comportement du détecteur sur des images sans visages."""
        blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
        faces = self.detector.detect(blank_img)
        self.assertIsNone(faces, "Le détecteur a trouvé un visage sur une image noire !")

        noise_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        # Ne doit pas planter
        annotated = self.detector.draw_faces(noise_img, None)
        self.assertEqual(annotated.shape, noise_img.shape)

    def test_03_feature_extraction_and_cosine_similarity(self):
        """Vérifie la normalisation, la forme (1, 128) et le calcul cosinus."""
        dummy_face_row = np.array(
            [50, 50, 100, 100, 75, 75, 125, 75, 100, 100, 80, 125, 120, 125, 0.99],
            dtype=np.float32,
        )
        dummy_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        aligned = self.encoder.recognizer.alignCrop(dummy_img, dummy_face_row)
        self.assertEqual(aligned.shape, (112, 112, 3), "L'alignement SFace doit produire une image 112x112")

        feat = self.encoder.recognizer.feature(aligned)
        self.assertEqual(feat.shape, (1, 128), "Le vecteur de caractéristiques doit être de dimension (1, 128)")

        # Similarité cosinus d'un vecteur avec lui-même = 1.0 (ou très proche)
        sim_self = self.encoder.compute_similarity(feat, feat)
        self.assertAlmostEqual(sim_self, 1.0, places=2)

    def test_04_encode_dataset_in_temp_dir_and_cleanup(self):
        """
        Vérifie l'encodage d'un jeu de données temporaire.
        Le dossier temporaire est automatiquement détruit après le test.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_dataset = temp_path / "mock_dataset"
            temp_encodings = temp_path / "mock_encodings.pkl"

            # Création de deux étudiants temporaires
            (temp_dataset / "student_temp1").mkdir(parents=True)
            (temp_dataset / "student_temp2").mkdir(parents=True)

            # Création d'images factices
            cv2.imwrite(str(temp_dataset / "student_temp1" / "img1.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
            cv2.imwrite(str(temp_dataset / "student_temp2" / "img2.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))

            # Exécution de l'encodage
            res = encode_dataset(
                dataset_dir=temp_dataset,
                output_file=temp_encodings,
                encoder=self.encoder,
            )

            # Vérification de la création du fichier pickle
            self.assertTrue(temp_encodings.exists(), "Le fichier pickle temporaire n'a pas été créé")
            self.assertIn("names", res)
            self.assertIn("encodings", res)
            self.assertIn("classes", res)

        # Vérification stricte : le dossier temporaire a bien été supprimé
        self.assertFalse(temp_path.exists(), "Le dossier temporaire doit être supprimé après le test")

    def test_05_unknown_person_threshold_logic(self):
        """
        Vérifie que :
        - Un vecteur identique ou très proche est bien reconnu avec son nom.
        - Un vecteur orthogonal / éloigné est bien rejeté et classé comme 'Inconnu'.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_encodings_file = Path(temp_dir) / "test_enc.pkl"

            # Création d'une base factice avec 1 identité connue
            base_vec = np.ones((1, 128), dtype=np.float32)
            base_vec = base_vec / np.linalg.norm(base_vec)

            data = {
                "names": ["student_01"],
                "encodings": [base_vec],
                "classes": ["student_01"],
            }
            with open(temp_encodings_file, "wb") as f:
                pickle.dump(data, f)

            recognizer = FaceRecognizer(
                encodings_file=temp_encodings_file,
                cosine_threshold=0.363,
                encoder=self.encoder,
                detector=self.detector,
            )

            # Test A : vecteur identique -> doit prédire "student_01"
            name_match, score_match = recognizer.identify_feature(base_vec)
            self.assertEqual(name_match, "student_01")
            self.assertGreaterEqual(score_match, 0.363)

            # Test C : vecteur opposé / orthogonal -> doit prédire "Inconnu"
            diff_vec = np.zeros((1, 128), dtype=np.float32)
            diff_vec[0, 0] = -1.0
            name_diff, score_diff = recognizer.identify_feature(diff_vec)
            self.assertEqual(name_diff, "Inconnu", "Une correspondance faible doit être rejetée en 'Inconnu'")

    def test_06_evaluate_module_with_temp_data(self):
        """Vérifie la génération du fichier CSV d'évaluation sans polluer results/."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_test_dir = temp_path / "test_data"
            temp_csv = temp_path / "test_results.csv"

            # Création de sous-dossiers de test
            (temp_test_dir / "student_01").mkdir(parents=True)
            cv2.imwrite(str(temp_test_dir / "student_01" / "sample.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))

            recognizer = FaceRecognizer(
                encodings_file=temp_path / "empty.pkl",
                cosine_threshold=0.363,
                encoder=self.encoder,
                detector=self.detector,
            )

            records = evaluate_dataset(recognizer, temp_test_dir, output_csv=temp_csv)
            self.assertTrue(temp_csv.exists(), "Le fichier CSV temporaire doit exister")
            self.assertEqual(len(records), 1)
            self.assertIn("test_id", records[0])
            self.assertIn("expected_identity", records[0])
            self.assertIn("predicted_identity", records[0])
            self.assertIn("confidence_score", records[0])
            self.assertIn("face_detected", records[0])
            self.assertIn("is_correct", records[0])

        # Nettoyage automatique validé
        self.assertFalse(temp_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
