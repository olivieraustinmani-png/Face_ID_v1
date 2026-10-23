"""
Module d'encodage des visages avec OpenCV SFace.

Rôles :
- Charger le modèle de reconnaissance SFace.
- Détecter les visages avec YuNet.
- Extraire les caractéristiques faciales.
- Encoder les images du dataset des étudiants.
- Sauvegarder les encodages dans un fichier pickle.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import cv2
import numpy as np

from src.detect_faces import FaceDetector


# ============================================================
# CHEMINS PAR DEFAUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_SFACE_MODEL = (
    BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
)

DEFAULT_DATASET_DIR = BASE_DIR / "dataset"

DEFAULT_ENCODINGS_FILE = (
    BASE_DIR / "models" / "encodings.pkl"
)


# ============================================================
# CONSTANTES OPENCV
# ============================================================

COSINE_METRIC = getattr(
    cv2,
    "FaceRecognizerSF_FR_COSINE",
    getattr(cv2.FaceRecognizerSF, "FR_COSINE", 0)
)


class FaceEncoder:
    """Encodeur facial utilisant le modèle SFace."""

    def __init__(
        self,
        sface_model_path: Optional[str | Path] = None,
        detector: Optional[FaceDetector] = None,
    ) -> None:

        # Déterminer le chemin du modèle.
        self.model_path = (
            Path(sface_model_path)
            if sface_model_path
            else DEFAULT_SFACE_MODEL
        )

        # Vérifier que le modèle existe.
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Modèle SFace introuvable : {self.model_path}"
            )

        # Charger le modèle de reconnaissance.
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.model_path),
                config=""
            )
        except cv2.error as exc:
            raise RuntimeError(
                "Impossible de charger le modèle SFace. "
                "Vérifie que le fichier ONNX est valide."
            ) from exc

        print("[INFO] Modèle SFace chargé avec succès.")

        # Réutiliser le détecteur fourni ou en créer un.
        self.detector = (
            detector if detector is not None else FaceDetector()
        )

        print("[INFO] Détecteur facial initialisé.")

    def extract_features(
        self,
        image: np.ndarray,
        faces: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Détecte les visages et extrait leurs caractéristiques.

        Retour :
            Liste de couples (informations_visage, embedding).
        """

        # Vérifier que l'image est exploitable.
        if image is None or not isinstance(image, np.ndarray):
            print("[ERREUR] Image invalide.")
            return []

        if image.size == 0:
            print("[ERREUR] Image vide.")
            return []

        if image.ndim != 3 or image.shape[2] != 3:
            print("[ERREUR] L'image doit être au format couleur BGR.")
            return []

        # Détecter les visages si les résultats ne sont pas fournis.
        if faces is None:
            try:
                faces = self.detector.detect(image)
            except Exception as exc:
                print(f"[ERREUR] Échec de la détection : {exc}")
                return []

        # Aucun visage trouvé.
        if faces is None or len(faces) == 0:
            return []

        results = []

        for index, face in enumerate(faces):

            try:
                # Aligner et recadrer le visage.
                aligned_face = self.recognizer.alignCrop(
                    image,
                    face
                )

                if aligned_face is None or aligned_face.size == 0:
                    print(
                        f"[ATTENTION] Alignement impossible "
                        f"pour le visage {index + 1}."
                    )
                    continue

                # Extraire l'embedding facial.
                feature = self.recognizer.feature(aligned_face)

                if feature is None or feature.size == 0:
                    print(
                        f"[ATTENTION] Caractéristiques absentes "
                        f"pour le visage {index + 1}."
                    )
                    continue

                # Vérifier que le vecteur contient des valeurs finies.
                if not np.isfinite(feature).all():
                    print(
                        f"[ATTENTION] Embedding invalide "
                        f"pour le visage {index + 1}."
                    )
                    continue

                results.append((face, feature))

            except cv2.error as exc:
                print(
                    f"[ATTENTION] Erreur OpenCV sur le visage "
                    f"{index + 1} : {exc}"
                )

        return results

    def compute_similarity(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """Calcule la similarité cosinus entre deux embeddings."""

        return float(
            self.recognizer.match(
                feat1,
                feat2,
                COSINE_METRIC
            )
        )


def encode_dataset(
    dataset_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    encoder: Optional[FaceEncoder] = None,
) -> Dict[str, Any]:
    """
    Parcourt le dataset et sauvegarde les embeddings obtenus.
    """

    dataset_path = (
        Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
    )

    output_path = (
        Path(output_file)
        if output_file
        else DEFAULT_ENCODINGS_FILE
    )

    # Vérifier l'existence du dataset.
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_path}"
        )

    # Initialiser l'encodeur.
    if encoder is None:
        encoder = FaceEncoder()

    valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    known_names: List[str] = []
    known_encodings: List[np.ndarray] = []
    stats: Dict[str, int] = {}

    total_images = 0
    unreadable_images = 0
    images_without_faces = 0
    images_with_faces = 0

    student_dirs = sorted(
        directory
        for directory in dataset_path.iterdir()
        if directory.is_dir()
    )

    print("\n" + "=" * 60)
    print("DÉMARRAGE DE L'ENCODAGE DU DATASET")
    print("=" * 60)

    print(f"[INFO] Dataset : {dataset_path}")
    print(f"[INFO] Dossiers d'étudiants : {len(student_dirs)}")

    if not student_dirs:
        print(
            "[ATTENTION] Aucun sous-dossier d'étudiant trouvé."
        )

    for student_dir in student_dirs:

        student_name = student_dir.name

        image_files = sorted(
            path
            for path in student_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in valid_extensions
        )

        stats[student_name] = 0

        print(
            f"\n[ÉTUDIANT] {student_name} "
            f"({len(image_files)} image(s))"
        )

        if not image_files:
            print(
                "[ATTENTION] Aucune image reconnue dans ce dossier."
            )

        for img_path in image_files:

            total_images += 1

            # Lire l'image en couleur.
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            
            # Détecter les visages.
            try:
                faces = encoder.detector.detect(image)
            except Exception as exc:
                print(
                    f"[ERREUR] Le détecteur a échoué "
                    f"sur {img_path.name} : {exc}"
                )
                continue

            print(
                f"[INFO] {len(faces)} visage(s) détecté(s)."
            )

            # Extraire les caractéristiques des visages détectés.
            results = encoder.extract_features(
                image,
                faces=faces
            )

            if not results:
                print(
                    "[ATTENTION] Visage détecté, "
                    "mais aucun embedding extrait."
                )
                continue

            # Sélectionner le visage ayant le meilleur score
            # de détection, situé à la dernière position du tableau.
            best_face, best_feature = max(
                results,
                key=lambda item: float(item[0][-1])
            )

            known_names.append(student_name)
            known_encodings.append(best_feature)
            stats[student_name] += 1
            images_with_faces += 1

            print(
                "[SUCCÈS] Embedding extrait : "
                f"forme {best_feature.shape}"
            )

    # Construire le dictionnaire final.
    data_to_save = {
        "names": known_names,
        "encodings": known_encodings,
        "classes": sorted(set(known_names)),
    }

    # Sauvegarder uniquement si des encodages ont été obtenus.
    if known_encodings:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "wb") as file:
            pickle.dump(data_to_save, file)

        print(
            f"\n[SUCCÈS] Encodages sauvegardés dans : "
            f"{output_path}"
        )

    else:
        print(
            "\n[ATTENTION] Aucun encodage obtenu. "
            "Le fichier existant n'a pas été remplacé."
        )

    # Afficher le bilan.
    print("\n" + "=" * 60)
    print("BILAN DE L'ENCODAGE")
    print("=" * 60)

    print(f"Images examinées : {total_images}")
    print(f"Images illisibles : {unreadable_images}")
    print(f"Images sans visage détecté : {images_without_faces}")
    print(f"Images avec embedding extrait : {images_with_faces}")
    print(f"Nombre total d'embeddings : {len(known_encodings)}")

    print("\nDétail par étudiant :")

    for name, count in stats.items():
        print(f"  - {name} : {count} embedding(s)")

    print("=" * 60 + "\n")

    return data_to_save


def load_encodings(
    encodings_file: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Charge les embeddings préalablement sauvegardés."""

    path = (
        Path(encodings_file)
        if encodings_file
        else DEFAULT_ENCODINGS_FILE
    )

    if not path.is_file():
        print(f"[INFO] Fichier d'encodages absent : {path}")
        return None

    with open(path, "rb") as file:
        data = pickle.load(file)

    print(
        f"[INFO] {len(data.get('encodings', []))} "
        "embedding(s) chargé(s)."
    )

    return data


if __name__ == "__main__":
    encode_dataset()
