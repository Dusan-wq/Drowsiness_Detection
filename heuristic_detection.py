from helpers import euclid
'''
    ---- Heuristicka detekcija ----
    Na osnovu parametara koje smo postavili mozemo da detektujemo umor, 
    bez ucitavanja slika ili videa za treniranje modela
'''
class Heuristic:
    def __init__(self):
        self._ear = 0
        self._mar = 0
        self._drowsy_frame_count = 0
        self._yawn_frame_count = 0
        self._head_pitch = 0

    # ---------- EAR ----------
    # Eye aspect ratio - govori nam o tome da li su oci zatvorene ili otvorene
    @property
    def ear(self):
        return self._ear

    @ear.setter
    def ear(self, value):
        self._ear = value

    # ---------- MAR ----------
    # Mouth aspect ratio - govori nam o tome da li je usta otvorena ili zatvorena
    @property
    def mar(self):
        return self._mar

    @mar.setter
    def mar(self, value):
        self._mar = value

    # ---------- Drowsy frame count ----------
    # Broji broj frame-ova koliko su oci zatvorene
    @property
    def drowsy_frame_count(self):
        return self._drowsy_frame_count

    @drowsy_frame_count.setter
    def drowsy_frame_count(self, value):
        self._drowsy_frame_count = value

    # ---------- Yawn frame count ----------
    # Broji broj frame-ova koliko je usta otvorena
    @property
    def yawn_frame_count(self):
        return self._yawn_frame_count

    @yawn_frame_count.setter
    def yawn_frame_count(self, value):
        self._yawn_frame_count = value

    # ---------- Head pitch ----------
    @property
    def head_pitch(self):
        return self._head_pitch

    @head_pitch.setter
    def head_pitch(self, value):
        self._head_pitch = value

    def eye_aspect_ratio(self,landmarks, idxs, img_w, img_h):
        pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idxs]
        A = euclid(pts[1], pts[5])
        B = euclid(pts[2], pts[4])
        C = euclid(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0
    
    def mouth_aspect_ratio(self,landmarks, idxs, img_w, img_h):
        pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idxs]
        v1 = euclid(pts[0], pts[1])
        v2 = euclid(pts[2], pts[3])
        h = euclid(pts[4], pts[5])
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0