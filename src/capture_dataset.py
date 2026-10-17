"""
Script interactif de collecte d'images pour le dataset des étudiants.
Permet de capturer les 20 à 40 photos par étudiant via la webcam
avec vérification en direct de la détection faciale.
"""

from pathlib import Path
import time
import cv2
import numpy as np

from src.detect_faces import FaceDetector

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def capture_student_images(student_name: str, target_count: int = 30) -> None:
    """
    Lance une session interactive de capture webcam pour un étudiant.

    Args:
        student_name: Nom ou identifiant de l'étudiant (ex: "student_01").
        target_count: Nombre cible d'images à capturer (défaut: 30).
    """
    save_dir = DATASET_DIR / student_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Décompte des images existantes pour ne pas écraser
    existing = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png"))
    current_idx = len(existing)

    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'accéder à la webcam.")
        return

    print("=" * 60)
    print(f" Session de capture pour : {student_name}")
    print(f" Destination : {save_dir}")
    print(f" Images existantes : {current_idx}")
    print(f" Cible recommandée : {target_count} images")
    print(" Commandes :")
    print("   [ESPACE] ou [C] : Prendre une photo")
    print("   [A]             : Mode rafale automatique (1 photo / 0.5s)")
    print("   [Q]             : Quitter et enregistrer")
    print("=" * 60)

    auto_mode = False
    last_auto_capture = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERREUR] Problème de lecture du flux webcam.")
                break

            h, w = frame.shape[:2]
            faces = detector.detect(frame)
            display = frame.copy()

            # Dessin de la détection faciale
            has_good_face = False
            if faces is not None and len(faces) > 0:
                has_good_face = True
                display = detector.draw_faces(display, faces)

            # Bandeau d'informations supérieur
            cv2.rectangle(display, (0, 0), (w, 55), (30, 30, 30), -1)
            status_text = f"{student_name} | Photos: {current_idx}/{target_count}"
            mode_text = "[AUTO]" if auto_mode else "[MANUEL]"
            cv2.putText(display, f"{status_text}  {mode_text}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Conseils de variations
            tips = [
                "Varier : face, 3/4, profil leger",
                "Varier : sourire, neutre, yeux ouverts/fermes",
                "Varier : distance (proche, eloigne)",
            ]
            tip_idx = (current_idx // 10) % len(tips)
            cv2.putText(display, f"Conseil: {tips[tip_idx]}", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Capture Dataset - Face_ID_v1", display)
            key = cv2.waitKey(30) & 0xFF

            now = time.time()
            trigger_capture = False

            if key in (ord('c'), ord('C'), 32):  # 32 = Espace
                trigger_capture = True
            elif key in (ord('a'), ord('A')):
                auto_mode = not auto_mode
                print(f"[MODE] Rafale automatique : {'ACTIVÉ' if auto_mode else 'DÉSACTIVÉ'}")
            elif key in (ord('q'), ord('Q'), 27):  # 27 = Echap
                print("[INFO] Fin de session demandée par l'utilisateur.")
                break

            if auto_mode and (now - last_auto_capture >= 0.6) and has_good_face:
                trigger_capture = True
                last_auto_capture = now

            if trigger_capture:
                if not has_good_face:
                    print("  [AVERTISSEMENT] Aucun visage net détecté, repositionnez-vous.")
                else:
                    current_idx += 1
                    filename = f"{student_name}_{current_idx:03d}.jpg"
                    filepath = save_dir / filename
                    # Sauvegarde de la trame brute
                    cv2.imwrite(str(filepath), frame)
                    print(f"  [+] Photo sauvegardee : {filename} ({current_idx}/{target_count})")

                    # Effet flash visuel
                    flash = np.full_like(display, 255)
                    cv2.imshow("Capture Dataset - Face_ID_v1", flash)
                    cv2.waitKey(50)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[RÉSUMÉ] Session terminée. Total de photos pour {student_name} : {current_idx}\n")


if __name__ == "__main__":
    import sys
    name = input("Entrez l'identifiant de l'étudiant (ex: student_01, student_02...) : ").strip()
    if not name:
        name = "student_01"
    capture_student_images(name)

