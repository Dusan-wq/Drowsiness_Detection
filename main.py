import cv2
import mediapipe as mp
import numpy as np
import time

from heuristic_detection import Heuristic
from helpers import euclid
from mediapipe.python.solutions import face_mesh_connections

if __name__ == '__main__':
    # ---------- PARAMETRI ----------
    if True:
        h_detection = Heuristic()

        # setovanje vrednosti
        h_detection.ear = 0.20
        h_detection.mar = 0.75
        h_detection.drowsy_frame_count = 10
        h_detection.yawn_frame_count = 5
        h_detection.head_pitch = 12.0

    # ---------- MEDIAPIPE ----------
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
   
    # Od svih tacaka na licu koje je face_mesh detektovao, za desno / levo oko i usta cemo posmattrati samo  ove
    DESNO_OKO_TACKE = [33, 160, 158, 133, 153, 144]
    LEVO_OKO_TACKE  = [263, 387, 385, 362, 380, 373]
    USTA_TACKE = [13, 14, 78, 308, 61, 291]

    # Video snimanje
    cap = cv2.VideoCapture(0)


    # brojači
    eye_closed_counter = 0
    yawn_counter = 0
    fatigue_state = False

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            # Dobijanje trenutnog framea sa kamere
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            visina_framea, sirina_framea = frame.shape[:2]

            results = face_mesh.process(frame_rgb)
            stanje_display = "BUDAN"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # EAR i MAR
                ear_r = h_detection.eye_aspect_ratio(landmarks, DESNO_OKO_TACKE, sirina_framea, visina_framea)
                ear_l = h_detection.eye_aspect_ratio(landmarks, LEVO_OKO_TACKE, sirina_framea, visina_framea)
                ear = (ear_l + ear_r) / 2.0

                mar = h_detection.mouth_aspect_ratio(landmarks, USTA_TACKE, sirina_framea, visina_framea)

                # Azuriraj broj zatvaranja ociju eye_closed_counter
                if ear < h_detection.ear:
                    eye_closed_counter += 1
                else:
                    eye_closed_counter = 0

                # Azuriraj broj zevanja yawn_counter
                if mar > h_detection.mar:
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                # logika umora
                # Moze se dodatno poboljsati ako se doda i head_pitc
                # Ovo and moze biti i or
                if eye_closed_counter >= h_detection.drowsy_frame_count and yawn_counter >= h_detection.yawn_frame_count:
                    stanje_display = "UMORAN"
                    fatigue_state = True
                else:
                    stanje_display = "BUDAN"
                    fatigue_state = False

                # crtanje face mesh kontura
                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_face_landmarks[0],
                    face_mesh_connections.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                # dodatni tekst
                cv2.putText(frame, f"Status: {stanje_display}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0,0,255) if fatigue_state else (0,255,0), 2)
                cv2.putText(frame, f"EAR: {ear:.2f} L:{ear_l:.2f} R:{ear_r:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1) 
                cv2.putText(frame, f"Eye closed frames: {eye_closed_counter}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"Yawn frames: {yawn_counter}", (10,135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            else:
                cv2.putText(frame, "NE VIDIM NIKAKVU FACU", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 2)

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
