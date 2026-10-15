"""
Module de reconnaissance faciale.
Ce module compare les visages détectés aux représentations enregistrées dans la base de données,
applique un seuil de décision et attribue l'identité de l'étudiant ou la mention « Inconnu ».
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from src.detect_faces import FaceDetector
from src.encode_faces import FaceEncoder, load_encodings, DEFAULT_ENCODINGS_FILE


class FaceRecognizer:
    """Système de reconnaissance et identification des visages."""

    def __init__(
        self,
        encodings_file: Optional[str | Path] = None,
        cosine_threshold: float = 0.363,
        encoder: Optional[FaceEncoder] = None,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        """
        Initialise le module de reconnaissance faciale.

        Args:
            encodings_file: Chemin vers la base d'encodages.
            cosine_threshold: Seuil de similarité cosinus (SFace standard: 0.363).
                              Au-dessus de ce seuil, le visage est accepté.
                              En-dessous, il est classé comme « Inconnu ».
            encoder: Instance de FaceEncoder.
            detector: Instance de FaceDetector.
        """
        self.encodings_file = Path(encodings_file) if encodings_file else DEFAULT_ENCODINGS_FILE
        self.cosine_threshold = cosine_threshold

        self.detector = detector if detector is not None else FaceDetector()
        self.encoder = encoder if encoder is not None else FaceEncoder(detector=self.detector)

        # Chargement de la base d'identités
        self.known_names: List[str] = []
        self.known_encodings: List[np.ndarray] = []
        self.reload_encodings()

    def reload_encodings(self) -> bool:
        """Recharge les encodages depuis le fichier disque."""
        data = load_encodings(self.encodings_file)
        if data is not None and "encodings" in data and len(data["encodings"]) > 0:
            self.known_names = data["names"]
            self.known_encodings = data["encodings"]
            print(f"[FaceRecognizer] Base chargée : {len(self.known_encodings)} visages de {len(set(self.known_names))} étudiants.")
            return True
        else:
            self.known_names = []
            self.known_encodings = []
            print(f"[FaceRecognizer] Attention : Aucune base d'encodages disponible à {self.encodings_file}.")
            return False

    def identify_feature(self, feature: np.ndarray) -> Tuple[str, float]:
        """
        Compare un vecteur de caractéristiques avec la base connue.

        Args:
            feature: Vecteur 128D d'un visage requête.

        Returns:
            Tuple[str, float]: (Nom ou "Inconnu", score_de_similarité).
        """
        if not self.known_encodings:
            return "Inconnu", 0.0

        best_score = -1.0
        best_name = "Inconnu"

        # Comparaison cosinus avec tous les vecteurs enregistrés
        for name, known_feat in zip(self.known_names, self.known_encodings):
            score = self.encoder.compute_similarity(feature, known_feat)
            if score > best_score:
                best_score = score
                best_name = name

        # Vérification du seuil de décision (Gestion des personnes inconnues)
        if best_score >= self.cosine_threshold:
            return best_name, max(0.0, min(1.0, best_score))
        else:
            return "Inconnu", max(0.0, min(1.0, best_score))

    def recognize_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte et identifie tous les visages présents dans une image / trame vidéo.

        Args:
            frame: Image BGR.

        Returns:
            List[Dict]: Liste des résultats pour chaque visage :
                        - box: (x, y, w, h)
                        - name: Identité prédite ("student_01", ..., "Inconnu")
                        - score: Score de similarité cosinus (0.0 à 1.0)
                        - is_known: bool
                        - landmarks: Coordonnées des 5 points clés
                        - face_raw: Données brutes YuNet
        """
        faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            # Alignement et extraction du vecteur 128D
            aligned_face = self.encoder.recognizer.alignCrop(frame, face)
            feature = self.encoder.recognizer.feature(aligned_face)

            # Identification et seuillage
            name, score = self.identify_feature(feature)

            box = face[0:4].astype(int)
            landmarks = face[4:14].reshape((5, 2)).astype(int)

            results.append({
                "box": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                "name": name,
                "score": float(score),
                "is_known": (name != "Inconnu"),
                "landmarks": landmarks,
                "face_raw": face,
            })

        return results

    def annotate_frame(
        self,
        frame: np.ndarray,
        results: List[Dict[str, Any]],
        show_landmarks: bool = True,
    ) -> np.ndarray:
        """
        Dessine les cadres, noms, scores et repères faciaux sur l'image.

        Args:
            frame: Image originale.
            results: Résultats renvoyés par recognize_frame.
            show_landmarks: Si True, affiche également les repères faciaux.

        Returns:
            np.ndarray: Image annotée.
        """
        output = frame.copy()

        for res in results:
            x, y, w, h = res["box"]
            name = res["name"]
            score = res["score"]
            is_known = res["is_known"]

            # Couleur : Vert si reconnu, Rouge si "Inconnu"
            color = (0, 220, 0) if is_known else (0, 0, 240)

            # Boîte de détection
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Bandeau de texte supérieur
            label = f"{name} ({score * 100:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bg_y1 = max(0, y - text_h - 10)
            bg_y2 = y
            cv2.rectangle(output, (x, bg_y1), (x + text_w + 10, bg_y2), color, -1)
            cv2.putText(
                output,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Repères faciaux
            if show_landmarks and "landmarks" in res:
                for lx, ly in res["landmarks"]:
                    cv2.circle(output, (lx, ly), 2, (255, 255, 0), -1)

        return output

