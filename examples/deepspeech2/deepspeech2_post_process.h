#ifndef __YOLOV5_POST_PROCESS_H__
#define __YOLOV5_POST_PROCESS_H__
#ifdef __cplusplus
       extern "C" {
#endif

int deepspeech2_post_process(float *tensor_data);

#ifdef __cplusplus
}
#endif

#endif
