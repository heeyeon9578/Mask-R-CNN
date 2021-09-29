# Mask R-CNN 을 이용하여 침실 내 객체 구별하기

## 핵심 코드는 main.ipynb 에 있습니다.

###### 저는 https://aihub.or.kr/ 사이트에 있는 "1인칭 시점 보행 영상" 데이터를 활용하였습니다. 데이터 중에 실내, 그중에 침실 데이터를 이용했습니다.
###### 위의 데이터를 이용하여 semantic segmentation 작업을 수행했습니다. 
###### custom.py 에 custom 데이터 이용시 바꿔야할 코드 사항이 있습니다. 
###### 모르는 사항이나 질문은 **heeyeon95781@gmail.com** 에 해주시길 바랍니다. 감사합니다.

### test 실행 후 결과 

'''python
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
'''
![Alt text](/path/to/img.jpg)
