"""
Module d'encodage des visages utilisant OpenCV SFace (Deep Learning).
Ce module extrait les représentations vectorielles (embeddings 128D) des visages
et permet d'encoder l'ensemble du jeu de données des étudiants.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import cv2
import numpy as np

from src.detect_faces import FaceDetector

# Chemins par défaut
DEFAULT_SFACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_recognition_sface_2021dec.onnx"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DEFAULT_ENCODINGS_FILE = Path(__file__).resolve().parent.parent / "models" / "encodings.pkl"

# Constante de comparaison cosinus compatible OpenCV 4.x et 5.x
COSINE_METRIC = getattr(cv2, "FaceRecognizerSF_FR_COSINE", getattr(cv2.FaceRecognizerSF, "FR_COSINE", 0))
L2_METRIC = getattr(cv2, "FaceRecognizerSF_FR_NORM_L2", getattr(cv2.FaceRecognizerSF, "FR_NORM_L2", 1))


class FaceEncoder:
    """Extracteur de caractéristiques faciales basé sur SFace."""

    def __init__(
        self,
        sface_model_path: Optional[str | Path] = None,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        """
        Initialise l'encodeur SFace et le détecteur YuNet.

        Args:
            sface_model_path: Chemin vers le modèle ONNX SFace.
            detector: Instance de FaceDetector existante (optionnel).
        """
        self.model_path = Path(sface_model_path) if sface_model_path else DEFAULT_SFACE_MODEL
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle SFace introuvable à l'emplacement : {self.model_path}.\n"
                "Veuillez vérifier que le fichier est présent dans le dossier models/."
            )

        self.recognizer = cv2.FaceRecognizerSF.create(model=str(self.model_path), config="")
        self.detector = detector if detector is not None else FaceDetector()

    def extract_features(
        self,
        image: np.ndarray,
        faces: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Détecte les visages (si non fournis) et extrait les vecteurs d'embeddings 128D.

        Args:
            image: Image BGR.
            faces: Visages déjà détectés par YuNet (optionnel).

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: Liste de paires (face_info, embedding_128d).
        """
        if faces is None:
            faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            # Alignement et recadrage normalisé (112x112)
            aligned_face = self.recognizer.alignCrop(image, face)
            # Extraction du vecteur de caractéristiques 128D
            feature = self.recognizer.feature(aligned_face)
            results.append((face, feature))

        return results

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs de caractéristiques.

        Args:
            feat1: Premier vecteur (1, 128).
            feat2: Second vecteur (1, 128).

        Returns:
            float: Score de similarité cosinus (plus la valeur est élevée, plus les visages sont proches).
        """
        return float(self.recognizer.match(feat1, feat2, COSINE_METRIC))


def encode_dataset(
    dataset_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    encoder: Optional[FaceEncoder] = None,
) -> Dict[str, Any]:
    """
    Parcourt le dossier dataset/, extrait les représentations faciales des étudiants
    et enregistre le dictionnaire d'encodages.

    Args:
        dataset_dir: Répertoire contenant les dossiers des étudiants.
        output_file: Fichier où sauvegarder les encodages sérialisés.
        encoder: Instance de FaceEncoder.

    Returns:
        Dict[str, Any]: Dictionnaire contenant les noms et les encodages correspondants.
    """
    dataset_path = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
    output_path = Path(output_file) if output_file else DEFAULT_ENCODINGS_FILE

    if encoder is None:
        encoder = FaceEncoder()

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    known_names: List[str] = []
    known_encodings: List[np.ndarray] = []
    stats: Dict[str, int] = {}

    student_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"\n[INFO] Démarrage de l'encodage du dataset : {dataset_path}")
    print(f"[INFO] {len(student_dirs)} dossiers d'étudiants trouvés.\n")

    for student_dir in student_dirs:
        student_name = student_dir.name
        image_files = [f for f in student_dir.iterdir() if f.suffix.lower() in valid_extensions]
        stats[student_name] = 0

        print(f" -> Traitement de '{student_name}' ({len(image_files)} images)...")

        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    [ATTENTION] Impossible de lire : {img_path.name}")
                continue

            results = encoder.extract_features(image)
            if not results:
                print(f"    [ATTENTION] Aucun visage détecté dans : {img_path.name}")
                continue

            # On prend le visage ayant le score de détection le plus élevé
            best_face, best_feature = max(results, key=lambda item: item[0][-1])
            known_names.append(student_name)
            known_encodings.append(best_feature)
            stats[student_name] += 1

    total_encodings = len(known_encodings)
    print(f"\n[INFO] Encodage terminé : {total_encodings} représentations extraites au total.")
    for name, count in stats.items():
        print(f"   * {name} : {count} visage(s) encodé(s)")

    data_to_save = {
        "names": known_names,
        "encodings": known_encodings,
        "classes": sorted(list(set(known_names))),
    }

    # Sauvegarde sur disque
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"[SUCCÈS] Encodages sauvegardés avec succès dans : {output_path}\n")
    return data_to_save


def load_encodings(encodings_file: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """
    Charge les encodages pré-calculés depuis le disque.

    Args:
        encodings_file: Chemin vers le fichier pickle des encodages.

    Returns:
        Dict ou None si le fichier n'existe pas.
    """
    path = Path(encodings_file) if encodings_file else DEFAULT_ENCODINGS_FILE
    if not path.exists():
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    encode_dataset()

