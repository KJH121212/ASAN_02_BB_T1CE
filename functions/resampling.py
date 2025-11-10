import SimpleITK as sitk

def resample_img(itk_image, out_size, is_label=False):

    original_spacing = itk_image.GetSpacing()       # (x,y,z) 방향의 물리적 거리
    original_size = itk_image.GetSize()             # (x,y,z) 방향의 voxel 개수

    # spacing 보정 (물리적 크기 유지)
    out_spacing = [
        original_spacing[0] * (original_size[0] / out_size[0]), # x 방향 보정
        original_spacing[1] * (original_size[1] / out_size[1]), # y 방향 보정
        original_spacing[2] * (original_size[2] / out_size[2])  # z 방향 보정 안함
    ]

    resample = sitk.ResampleImageFilter()                           # resampling filter 생성
    resample.SetOutputSpacing(out_spacing)                          # 새로운 voxel간 거리 지정
    resample.SetSize(out_size)                                      # 새로운 voxel 개수 지정
    resample.SetOutputDirection(itk_image.GetDirection())           # 방향 행렬 유지 (영상의 회전 정보)
    resample.SetOutputOrigin(itk_image.GetOrigin())                 # 원점 좌표 유지 (물리적 위치 보존)
    resample.SetTransform(sitk.Transform())                         # 변환 없음 (Identity transfom)
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())      # 기본 pixel 값 설정 (type 유지)

    # 영상/라벨 타입에 따라 보간법 결정
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)

    return resample.Execute(itk_image)