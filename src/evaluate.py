"""
Module d'évaluation et de benchmark du système de reconnaissance faciale.
Génère le rapport de test dans results/test_results.csv conformément à la section 10 du README.
"""

from pathlib import Path
from typing import Dict, List, Optional
import csv
import cv2
import pandas as pd

from src.recognize_faces import FaceRecognizer

RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "test_results.csv"


def evaluate_dataset(
    recognizer: FaceRecognizer,
    test_dir: Path,
    output_csv: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """
    Évalue les performances du système sur un ensemble d'images de test.

    Colonnes enregistrées :
    - test_id
    - expected_identity
    - predicted_identity
    - confidence_score
    - face_detected
    - is_correct
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    records: List[Dict] = []
    test_id = 1

    print(f"\n[INFO] Démarrage de l'évaluation sur : {test_dir}")

    # Parcourt les sous-dossiers (chaque sous-dossier correspond à l'identité attendue)
    for folder in sorted(test_dir.iterdir()):
        if not folder.is_dir():
            continue

        expected_identity = folder.name  # ex: "student_01" ou "inconnu"
        images = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]

        for img_path in images:
            image = cv2.imread(str(img_path))
            face_detected = False
            predicted = "Aucun_visage"
            score = 0.0

            if image is not None:
                results = recognizer.recognize_frame(image)
                if results:
                    face_detected = True
                    # On retient le visage principal
                    main_face = max(results, key=lambda r: r["box"][2] * r["box"][3])
                    predicted = main_face["name"]
                    score = main_face["score"]

            # Règle de vérification
            # Si attendu est un étudiant connu, predicted doit correspondre
            # Si attendu est "inconnu" ou absent de la base, predicted doit être "Inconnu"
            if expected_identity.lower() in ["inconnu", "unknown"]:
                is_correct = (predicted.lower() == "inconnu")
            else:
                is_correct = (predicted == expected_identity)

            records.append({
                "test_id": f"TEST_{test_id:04d}",
                "image_name": img_path.name,
                "expected_identity": expected_identity,
                "predicted_identity": predicted,
                "confidence_score": round(score, 4),
                "face_detected": face_detected,
                "is_correct": is_correct,
            })
            test_id += 1

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(output_csv, index=False, encoding="utf-8")
        accuracy = (df["is_correct"].sum() / len(df)) * 100
        print(f"\n[RÉSULTAT DE L'ÉVALUATION]")
        print(f"Total des tests : {len(df)}")
        print(f"Visages détectés : {df['face_detected'].sum()} / {len(df)}")
        print(f"Prédictions correctes : {df['is_correct'].sum()} / {len(df)}")
        print(f"Précision globale (Accuracy) : {accuracy:.2f}%")
        print(f"Rapport sauvegardé dans : {output_csv}\n")
    else:
        print("[ATTENTION] Aucune image de test trouvée.")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Évaluation de Face_ID_v1")
    parser.add_argument("--test-dir", type=str, default="dataset", help="Dossier contenant les images à tester")
    parser.add_argument("--threshold", type=float, default=0.363, help="Seuil de similarité cosinus")
    args = parser.parse_args()

    rec = FaceRecognizer(cosine_threshold=args.threshold)
    evaluate_dataset(rec, Path(args.test_dir))

