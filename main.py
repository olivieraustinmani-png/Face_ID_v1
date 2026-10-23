"""
Point d'entrée principal de l'application Face_ID_v1.
Reconnaissance faciale intelligente des étudiants en temps réel ou sur image fixe.
"""

from datetime import datetime
from pathlib import Path
import argparse
import sys
import time
import cv2
import numpy as np
import json

from src.detect_faces import FaceDetector
from src.encode_faces import FaceEncoder, encode_dataset, DEFAULT_ENCODINGS_FILE, DEFAULT_DATASET_DIR
from src.recognize_faces import FaceRecognizer
from src.capture_dataset import capture_student_images

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results" / "screenshots"

def load_student_names():
    names_file = BASE_DIR / "students.json"

    if not names_file.exists():
        print("[ATTENTION] Le fichier students.json est introuvable.")
        return {}

    try:
        with names_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            print("[ERREUR] students.json doit contenir un objet JSON.")
            return {}

        return data

    except json.JSONDecodeError:
        print("[ERREUR] Le fichier students.json contient un JSON invalide.")
        return {}

def apply_student_names(results, student_names):
    """Remplace les identifiants techniques par les noms à afficher."""

    for result in results:
        student_id = result.get("name", "Inconnu")

        if student_id not in ("Inconnu", "Unknown", None):
            result["name"] = student_names.get(
                student_id,
                student_id
            )

    return results

def run_webcam(recognizer: FaceRecognizer, camera_index: int = 0) -> None:
    """Lance la reconnaissance faciale en temps réel sur le flux webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la webcam (index {camera_index}).")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print(" RECONNAISSANCE FACIALE EN TEMPS RÉEL (WEBCAM)")
    print(" Commandes :")
    print("   [S] : Sauvegarder une capture d'écran dans results/screenshots/")
    print("   [Q] ou [ÉCHAP] : Quitter l'application")
    print("=" * 65 + "\n")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERREUR] Erreur de lecture du flux vidéo.")
                break

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (now - prev_time)))
            prev_time = now

            # Détection et identification
            results = recognizer.recognize_frame(frame)

            student_names = load_student_names()
            results = apply_student_names(results, student_names)

            annotated = recognizer.annotate_frame(frame, results)

            # Informations d'état et FPS
            h, w = frame.shape[:2]
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Etudiants en base: {len(set(recognizer.known_names))} | Seuil: {recognizer.cosine_threshold:.2f}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Face_ID_v1 - Reconnaissance Faciale", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("[INFO] Arrêt de la webcam.")
                break
            elif key in (ord('s'), ord('S')):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = RESULTS_DIR / f"capture_{timestamp}.png"
                cv2.imwrite(str(save_path), annotated)
                print(f"[CAPTURE] Image sauvegardée dans : {save_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(recognizer: FaceRecognizer, image_path: Path) -> None:
    """Effectue la détection et la reconnaissance sur une image statique."""
    if not image_path.exists():
        print(f"[ERREUR] Fichier introuvable : {image_path}")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERREUR] Impossible de charger l'image : {image_path}")
        return

    print(f"\n[INFO] Analyse de l'image : {image_path.name}")
    results = recognizer.recognize_frame(frame)

    student_names = load_student_names()
    results = apply_student_names(results, student_names)

    annotated = recognizer.annotate_frame(frame, results)

    print(f"[RÉSULTAT] {len(results)} visage(s) détecté(s) :")
    for idx, res in enumerate(results, start=1):
        x, y, w, h = res["box"]
        status = "RECONNU" if res["is_known"] else "NON RECONNU"
        print(f"  {idx}. Identité : {res['name']:<12} | Score : {res['score'] * 100:.1f}% | Statut : {status}")

    # Sauvegarde automatique du résultat annoté
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"result_{image_path.stem}.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"[INFO] Résultat annoté sauvegardé dans : {out_path}")

    # Affichage de l'image
    cv2.imshow(f"Face_ID_v1 - {image_path.name} (Appuyez sur une touche)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interactive_menu(recognizer: FaceRecognizer) -> None:
    """Affiche un menu en ligne de commande pour guider l'utilisateur."""
    while True:
        print("\n" + "=" * 55)
        print("          FACE_ID_v1 - RECONNAISSANCE FACIALE")
        print("=" * 55)
        print(" 1. Lancer la reconnaissance via la webcam")
        print(" 2. Analyser une image sur le disque")
        print(" 3. Enregistrer / Mettre à jour la base des visages (dataset)")
        print(" 4. Capturer des photos pour un étudiant (webcam)")
        print(" 5. Modifier le seuil de décision (actuel : {:.2f})".format(recognizer.cosine_threshold))
        print(" 0. Quitter")
        print("=" * 55)

        choice = input("Votre choix (0-5) : ").strip()

        if choice == "1":
            if not recognizer.known_encodings:
                print("\n[ATTENTION] La base d'encodages est vide !")
                print("Veuillez d'abord ajouter des photos dans 'dataset/' et lancer l'option 3.")
                confirm = input("Voulez-vous quand même tester la détection ? (o/n) : ").lower().strip()
                if confirm != "o":
                    continue
            run_webcam(recognizer)

        elif choice == "2":
            img_input = input("Entrez le chemin absolu ou relatif de l'image : ").strip().strip('"')
            if img_input:
                run_image(recognizer, Path(img_input))

        elif choice == "3":
            print("\n[INFO] Encodage du dossier dataset/ en cours...")
            encode_dataset(DEFAULT_DATASET_DIR, DEFAULT_ENCODINGS_FILE)
            recognizer.reload_encodings()

        elif choice == "4":
            student = input(
                "Identifiant de l'étudiant (ex: student_01) : "
            ).strip()

            if student:
                count_str = input(
                    "Nombre d'images à capturer (défaut : 30) : "
                ).strip()

                count = int(count_str) if count_str.isdigit() else 30

                if count < 1:
                    print("[ERREUR] Le nombre d'images doit être supérieur à zéro.")
                    continue

                capture_student_images(student, count)

                reload_prompt = input(
                    "Souhaitez-vous mettre à jour la base d'encodages maintenant ? (o/n) : "
                ).lower().strip()

                if reload_prompt == "o":
                    encode_dataset(
                        DEFAULT_DATASET_DIR,
                        DEFAULT_ENCODINGS_FILE
                    )
                    recognizer.reload_encodings()

        elif choice == "5":
            val = input(f"Nouveau seuil de similarité cosinus (défaut 0.363, actuel {recognizer.cosine_threshold}) : ").strip()
            try:
                new_thresh = float(val)
                recognizer.cosine_threshold = new_thresh
                print(f"[SUCCÈS] Seuil mis à jour à : {new_thresh}")
            except ValueError:
                print("[ERREUR] Valeur numérique invalide.")

        elif choice in ("0", "q", "exit"):
            print("\nAu revoir !")
            break
        else:
            print("[ERREUR] Option inconnue, réessayez.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Application Face_ID_v1 - Reconnaissance faciale intelligente."
    )

    parser.add_argument(
        "--mode",
        choices=["webcam", "image", "encode", "capture", "menu"],
        default="menu",
        help="Mode d'exécution de l'application."
    )

    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Chemin de l'image (mode image)."
    )

    parser.add_argument(
        "--student",
        type=str,
        default="student_01",
        help="Identifiant de l'étudiant (mode capture)."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Nombre d'images à capturer."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.363,
        help="Seuil de similarité cosinus."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index de la caméra."
    )

    args = parser.parse_args()

    recognizer = FaceRecognizer(
        cosine_threshold=args.threshold
    )

    if args.mode == "encode":
        encode_dataset(
            DEFAULT_DATASET_DIR,
            DEFAULT_ENCODINGS_FILE
        )
        recognizer.reload_encodings()

    elif args.mode == "capture":
        if args.count < 1:
            print("[ERREUR] Le nombre d'images doit être supérieur à zéro.")
            sys.exit(1)

        capture_student_images(args.student, args.count)

    elif args.mode == "image":
        if not args.image:
            print("[ERREUR] Veuillez spécifier une image avec --image.")
            sys.exit(1)

        run_image(recognizer, Path(args.image))

    elif args.mode == "webcam":
        run_webcam(recognizer, camera_index=args.camera)

    else:
        interactive_menu(recognizer)


if __name__ == "__main__":
    main()
