"""
Module de détection faciale utilisant OpenCV YuNet.
Ce module fournit la détection de visages en temps réel sur images et flux vidéo.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Chemins par défaut
DEFAULT_YUNET_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    """Détecteur de visages basé sur le modèle ONNX YuNet d'OpenCV."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        """
        Initialise le détecteur YuNet.

        Args:
            model_path: Chemin vers le fichier ONNX du modèle YuNet.
            conf_threshold: Seuil de confiance minimal pour retenir une détection.
            nms_threshold: Seuil de suppression des non-maximaux.
            top_k: Nombre maximum de visages conservés avant NMS.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_YUNET_MODEL
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle YuNet introuvable à l'emplacement : {self.model_path}.\n"
                "Veuillez vérifier que le fichier est présent dans le dossier models/."
            )

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._current_input_size = (320, 320)

        # Création de l'instance OpenCV FaceDetectorYN
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=self._current_input_size,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def _update_input_size(self, width: int, height: int) -> None:
        """Met à jour la taille d'entrée du détecteur si les dimensions changent."""
        if self._current_input_size != (width, height):
            self._current_input_size = (width, height)
            self.detector.setInputSize((width, height))

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Détecte les visages dans une image BGR.

        Args:
            image: Image sous forme de tableau NumPy (BGR).

        Returns:
            np.ndarray: Tableau de forme (N, 15) représentant les N visages détectés,
                        ou None si aucun visage n'est détecté.
                        Chaque ligne contient :
                        [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        self._update_input_size(w, h)

        # OpenCV FaceDetectorYN.detect retourne (status, faces)
        _, faces = self.detector.detect(image)
        return faces

    @staticmethod
    def draw_faces(
        image: np.ndarray,
        faces: Optional[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Dessine les boîtes englobantes et les repères faciaux sur une copie de l'image.

        Args:
            image: Image d'origine (BGR).
            faces: Visages retournés par detect().
            color: Couleur BGR du cadre.
            thickness: Épaisseur des traits.

        Returns:
            np.ndarray: Image annotée.
        """
        output = image.copy()
        if faces is None:
            return output

        for face in faces:
            # Boîte englobante
            box = face[0:4].astype(int)
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

            # Score de détection
            score = face[-1]
            label = f"Visage: {score:.1%}"
            cv2.putText(
                output,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            # Landmarks (yeux, nez, coins de la bouche)
            landmarks = face[4:14].reshape((5, 2)).astype(int)
            landmark_colors = [
                (255, 0, 0),    # oeil droit
                (0, 0, 255),    # oeil gauche
                (0, 255, 255),  # nez
                (255, 255, 0),  # coin droit bouche
                (0, 255, 0),    # coin gauche bouche
            ]
            for (lx, ly), c in zip(landmarks, landmark_colors):
                cv2.circle(output, (lx, ly), 2, c, -1)

        return output


if __name__ == "__main__":
    import sys

    print("Initialisation du détecteur YuNet...")
    detector = FaceDetector()
    print("Détecteur initialisé avec succès !")

    # Test avec la webcam si disponible
    print("Ouverture du flux vidéo (webcam)... Appuyez sur 'q' pour quitter.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'accéder à la webcam. Test avec une image synthétique...")
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(blank_img)
        print("Test détection image synthétique réussi (0 visage attendu).")
        sys.exit(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detector.detect(frame)
            annotated = detector.draw_faces(frame, faces)
            cv2.imshow("Detection Faciale - YuNet (Q pour quitter)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

