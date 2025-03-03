# BRISK 알고리즘 기반 특징점 매칭
import sys
import numpy as np
import cv2

# 사진 불러오기
img1 = r'C:\Users\설재훈\OneDrive\바탕 화면\Spyder\Source\Test\BLD00001_PS3_K3A_NIA0276.png'
img2 = r'C:\Users\설재훈\OneDrive\바탕 화면\Spyder\Source\Test\BLD00001_PS3_K3A_NIA0276(3).png'

# 파일을 바이너리 모드로 읽기
with open(img1, 'rb') as f:
    img_array1 = np.asarray(bytearray(f.read()), dtype=np.uint8)
with open(img2, 'rb') as f:
    img_array2 = np.asarray(bytearray(f.read()), dtype=np.uint8)

# openCV로 이미지 로드
src1 = cv2.imdecode(img_array1, cv2.IMREAD_GRAYSCALE)
src2 = cv2.imdecode(img_array2, cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print("Image load failed!")
    sys.exit()
    
# BRISK 알고리즘 객체 생성
feature = cv2.BRISK_create()

# 특징점 검출 및 기술자 계산
kp1, desc1 = feature.detectAndCompute(src1, None)
kp2, desc2 = feature.detectAndCompute(src2, None)

# BFMatcher 기반 특징점 매칭
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)

# 굳 매칭 결과 선별
good_matches = matches[:int(len(matches) * 0.5)]

print("Original Keypoint:", len(kp1))
print("Piece Keypoint:", len(kp2))
print("Total matches:", len(matches))
print("Good matches:", len(good_matches))

# 매칭된 좌표 추출
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 호모그래피 계산
if len(good_matches) > 4:  # 최소 4개의 매칭점 필요
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 10.0)

    # 잘린 이미지의 경계를 원본 이미지에 매핑
    h, w = src2.shape
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)  # 잘린 이미지의 모서리 좌표
    dst = cv2.perspectiveTransform(pts, H)  # 호모그래피 변환

    # 매칭 결과에 초록색 경계선 그리기
    result = cv2.cvtColor(src1, cv2.COLOR_GRAY2BGR)
    cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 3)
    
    # 매칭 개수 측정
    matchesMask = mask.ravel().tolist()
    TP = matchesMask.count(1)  # 올바른 매칭 개수
    FP = matchesMask.count(0)  # 잘못된 매칭 개수
    
    # 정확도 계산
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
else:
    print("Not enough good matches found to compute homography.")
    result = src1

# 특징점 매칭 결과 영상 생성
match_img = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
cv2.namedWindow('Matched Features', cv2.WINDOW_NORMAL)
cv2.namedWindow('Detected Region', cv2.WINDOW_NORMAL)
cv2.imshow("Matched Features", match_img)
cv2.imshow("Detected Region", result)
cv2.waitKey()
cv2.destroyAllWindows()
    
