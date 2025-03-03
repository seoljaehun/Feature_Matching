# SIFT 알고리즘 기반 특징점 매칭
import sys
import numpy as np
import cv2

# 사진 불러오기
img1 = r'C:\Users\설재훈\OneDrive\바탕 화면\Spyder\Source\Test\BLD00001_PS3_K3A_NIA0276.png'
img2 = r'C:\Users\설재훈\OneDrive\바탕 화면\Spyder\Source\Test\BLD00001_PS3_K3A_NIA0276(3).png'
#img2 = r'C:\Users\설재훈\OneDrive\바탕 화면\Spyder\Source\localization\1_14.png'

# 파일을 바이너리 모드로 읽기
with open(img1, 'rb') as f:
    img_array1 = np.asarray(bytearray(f.read()), dtype=np.uint8)
with open(img2, 'rb') as f:
    img_array2 = np.asarray(bytearray(f.read()), dtype=np.uint8)

# openCV로 이미지 로드 (흑백 변환)
src1 = cv2.imdecode(img_array1, cv2.IMREAD_GRAYSCALE)
src2 = cv2.imdecode(img_array2, cv2.IMREAD_GRAYSCALE)

# 양방향 필터 (노이즈 제거)
src1 = cv2.bilateralFilter(src1, -1, 10, 5)
src2 = cv2.bilateralFilter(src2, -1, 10, 5)

# CLAHE 적용 (대비 향상)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
src1 = clahe.apply(src1)
src2 = clahe.apply(src2)

# 히스토그램 매칭 함수
def histogram_matching(src2, src1):
    src2_hist, _ = np.histogram(src2.flatten(), 256, [0, 256])
    src1_hist, _ = np.histogram(src1.flatten(), 256, [0, 256])
    
    src_cdf = src2_hist.cumsum() / src2_hist.sum()
    ref_cdf = src1_hist.cumsum() / src1_hist.sum()
    
    lookup_table = np.searchsorted(ref_cdf, src_cdf).astype(np.uint8)
    matched = lookup_table[src2]
    return matched

src2 = histogram_matching(src2, src1)

# 가우시안 블러 적용 (노이즈 제거) + 언샤프 마스크 필터 (엣지 강조)
#blr1 = cv2.GaussianBlur(src1, (5,5), 0)
#blr2 = cv2.GaussianBlur(src2, (5,5), 0)
#src1 = cv2.addWeighted(src1, 1.5, blr1, -0.5, 0)
#src2 = cv2.addWeighted(src2, 1.5, blr2, -0.5, 0)

# 히스토그램 정규화 (화질 개선)
#src1 = cv2.normalize(src1, None, 0, 255, cv2.NORM_MINMAX)
#src2 = cv2.normalize(src2, None, 0, 255, cv2.NORM_MINMAX)

if src1 is None or src2 is None:
    print("Image load failed!")
    sys.exit()

# SIFT 알고리즘 객체 생성
feature = cv2.SIFT_create()

# 특징점 검출 및 기술자 계산
kp1, desc1 = feature.detectAndCompute(src1, None)
kp2, desc2 = feature.detectAndCompute(src2, None)

# FLANN 기반 특징점 매칭
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 검색 횟수 조정

matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(desc1, desc2, k=2)

# 굳 매칭 결과 선별
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

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
