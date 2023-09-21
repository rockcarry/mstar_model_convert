#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MODEL_INPUT_WIDTH   640
#define MODEL_INPUT_HEIGHT  384
#define MODEL_CLASS_NUMBER  80
#define MODEL_OUTPUT_NUM    2
#define MAX_BBOX_NUM        200
#define SCORE_THRESH        0.6
#define NMSIOU_THRESH       0.3
#define CLIP(x, y) ((x) < 0 ? 0 : ((x) > (y) ? (y) : (x)))

typedef struct {
    int   type;
    float score, x1, y1, x2, y2;
} BBOX;

static float s_anchor_list[MODEL_OUTPUT_NUM][6] = {
    { 115, 73, 119, 199, 242, 238 },
    { 12 , 18, 37 , 49 , 52 , 132 },
};

static int load_output_data(char *file, int idx, float **data, int *h, int *w, int *c)
{
    FILE *fp   = NULL;
    char *fbuf = NULL;
    char *pstart, *pcur;
    int   dim = 0, fsize, ok = 0, tmp, i;

    *data = NULL; // set data to NULL

    fp = fopen(file, "rb");
    if (!fp) {
        printf("failed to open sgs model output file: %s\n", file);
        goto done;
    }

    fseek(fp, 0, SEEK_END);
    fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    fbuf  = malloc(fsize);
    if (!fbuf) {
        printf("failed to allocate buffer for file read, file size: %d !\n", fsize);
        goto done;
    }
    fread(fbuf, 1, fsize, fp);

    pstart = fbuf;
    for (i=0; i<=idx; i++) {
        pcur   = strstr(pstart, "tensor dim:");
        pstart = pcur + strlen("tensor dim:");
    }

    sscanf(pstart, "%d", &dim);
    pcur  = strstr(pstart, "Original shape:[");
    pcur += strlen("Original shape:[");
    if (dim == 3) {
        sscanf(pcur, "%d %d %d", h, w, c);
        if (*h <= 0 || *w <= 0 || *c <= 0) {
            printf("invalid h/w/c value ! h: %d, w: %d, c: %d\n", *h, *w, *c);
            goto done;
        } else {
            printf("output sensor idx: %d, h: %d, w: %d, c: %d\n", idx, *h, *w, *c);
        }
    } else if (dim == 4) {
        sscanf(pcur, "%d %d %d %d", &tmp, h, w, c);
        if (*h <= 0 || *w <= 0 || *c <= 0) {
            printf("invalid h/w/c value ! h: %d, w: %d, c: %d\n", *h, *w, *c);
            goto done;
        } else {
            printf("output sensor idx: %d, h: %d, w: %d, c: %d\n", idx, *h, *w, *c);
        }
    } else {
        printf("invalid dim value %d !\n", dim);
        goto done;
    }

    *data = (float*)malloc(*h * *w * *c * sizeof(float));
    if (!*data) {
        printf("failed to allocate buffer for data !\n");
        goto done;
    }

    pcur = strstr(pcur, "tensor data:");
    if (!pcur) {
        printf("failed to find tensor data !\n");
        goto done;
    }

    fseek(fp, pcur - fbuf + strlen("tensor data:"), SEEK_SET);
    for (i=0; i<*h**w**c; i++) fscanf(fp, "%f", *data + i);
    ok = 1;

done:
    if (!ok ) { free(*data); *data = NULL; }
    if (fp  ) fclose(fp  );
    if (fbuf) free  (fbuf);
    return ok ? 0 : -1;
}

static void  free_output_data(float *data) { free(data); }
static float sigmoid(float x) { return 1.0f / (1.0f + (float)exp(-x)); }
static float get_layer_data(float *data, int h, int w, int c, int y, int x, int i) { return data[y * w * c + x * c + i]; }

static int yolov3(float *data, int h, int w, int c, int class_num, float score_thres, float anchor[6], BBOX *bbox_list, int bbox_cursize, int bbox_maxsize)
{
    int i, j, k, l; float score;
    for (i=0; i<h; i++) {
        for (j=0; j<w; j++) {
            for (k=0; k<3; k++) {
                int dstart = k * (4 + 1 + class_num), cindex = 0;
                float bs = get_layer_data(data, h, w, c, i, j, dstart + 4);
                float cs = get_layer_data(data, h, w, c, i, j, dstart + 5);
                for (l=1; l<class_num; l++) {
                    float val = get_layer_data(data, h, w, c, i, j, dstart + 5 + l);
                    if (cs < val) { cs = val; cindex = l; }
                }
                score = 1.0f / ((1.0f + (float)exp(-bs) * (1.0f + (float)exp(-cs))));
                if (score >= score_thres) {
                    float tx = get_layer_data(data, h, w, c, i, j, dstart + 0);
                    float ty = get_layer_data(data, h, w, c, i, j, dstart + 1);
                    float tw = get_layer_data(data, h, w, c, i, j, dstart + 2);
                    float th = get_layer_data(data, h, w, c, i, j, dstart + 3);
                    float bbox_cx = (j + sigmoid(tx)) * MODEL_INPUT_WIDTH  / w;
                    float bbox_cy = (i + sigmoid(ty)) * MODEL_INPUT_HEIGHT / h;
                    float bbox_w  = (float)exp(tw) * anchor[k * 2 + 0];
                    float bbox_h  = (float)exp(th) * anchor[k * 2 + 1];
                    if (bbox_cursize < bbox_maxsize) {
                        bbox_list[bbox_cursize].type  = cindex;
                        bbox_list[bbox_cursize].score = score;
                        bbox_list[bbox_cursize].x1    = bbox_cx - bbox_w * 0.5f;
                        bbox_list[bbox_cursize].y1    = bbox_cy - bbox_h * 0.5f;
                        bbox_list[bbox_cursize].x2    = bbox_cx + bbox_w * 0.5f;
                        bbox_list[bbox_cursize].y2    = bbox_cy + bbox_h * 0.5f;
                        bbox_cursize++;
                    }
                }
            }
        }
    }
    return bbox_cursize;
}

static int bbox_cmp(const void *p1, const void *p2)
{
    if      (((BBOX*)p1)->score < ((BBOX*)p2)->score) return  1;
    else if (((BBOX*)p1)->score > ((BBOX*)p2)->score) return -1;
    else return 0;
}

static int nms(BBOX *bboxlist, int n, float threshold, int min, int picw, int pich)
{
    int i, j, c;
    if (!bboxlist || !n) return 0;
    qsort(bboxlist, n, sizeof(BBOX), bbox_cmp);
    for (i=0; i<n && i!=-1; ) {
        for (c=i,j=i+1,i=-1; j<n; j++) {
            if (bboxlist[j].score == 0) continue;
            if (bboxlist[c].type == bboxlist[j].type) {
                float xc1, yc1, xc2, yc2, sc, s1, s2, ss, iou;
                xc1 = bboxlist[c].x1 > bboxlist[j].x1 ? bboxlist[c].x1 : bboxlist[j].x1;
                yc1 = bboxlist[c].y1 > bboxlist[j].y1 ? bboxlist[c].y1 : bboxlist[j].y1;
                xc2 = bboxlist[c].x2 < bboxlist[j].x2 ? bboxlist[c].x2 : bboxlist[j].x2;
                yc2 = bboxlist[c].y2 < bboxlist[j].y2 ? bboxlist[c].y2 : bboxlist[j].y2;
                sc  = (xc1 < xc2 && yc1 < yc2) ? (xc2 - xc1) * (yc2 - yc1) : 0;
                s1  = (bboxlist[c].x2 - bboxlist[c].x1) * (bboxlist[c].y2 - bboxlist[c].y1);
                s2  = (bboxlist[j].x2 - bboxlist[j].x1) * (bboxlist[j].y2 - bboxlist[j].y1);
                ss  = s1 + s2 - sc;
                if (min) iou = sc / (s1 < s2 ? s1 : s2);
                else     iou = sc / ss;
                if (iou > threshold) bboxlist[j].score = 0;
                else if (i == -1) i = j;
            } else if (i == -1) i = j;
        }
    }
    for (i=0,j=0; i<n; i++) {
        if (bboxlist[i].score) {
            bboxlist[j  ].score= bboxlist[i].score;
            bboxlist[j  ].type = bboxlist[i].type;
            bboxlist[j  ].x1   = bboxlist[i].x1 * picw / MODEL_INPUT_WIDTH ;
            bboxlist[j  ].y1   = bboxlist[i].y1 * pich / MODEL_INPUT_HEIGHT;
            bboxlist[j  ].x2   = bboxlist[i].x2 * picw / MODEL_INPUT_WIDTH ;
            bboxlist[j++].y2   = bboxlist[i].y2 * pich / MODEL_INPUT_HEIGHT;
        }
    }
    return j;
}

static float* ufface_init_priors(int inputw, int inputh, int *num_priors)
{
    float STRIDES[] = { 8.0f, 16.0f, 32.0f, 64.0f };
    float SHRINKAGES[][4] = {
        { STRIDES[0], STRIDES[1], STRIDES[2], STRIDES[3] },
        { STRIDES[0], STRIDES[1], STRIDES[2], STRIDES[3] },
    };
    float MINBOXES[][3] = {
        { 10.0f ,  16.0f , 24.0f  },
        { 32.0f ,  48.0f , 0      },
        { 64.0f ,  96.0f , 0      },
        { 128.0f,  192.0f, 256.0f },
    };
    float featuremap_size[2][4] = {
        { (float)ceil(inputw / STRIDES[0]), (float)ceil(inputw / STRIDES[1]), (float)ceil(inputw / STRIDES[2]), (float)ceil(inputw / STRIDES[3]) },
        { (float)ceil(inputh / STRIDES[0]), (float)ceil(inputh / STRIDES[1]), (float)ceil(inputh / STRIDES[2]), (float)ceil(inputh / STRIDES[3]) },
    };

    int    i, j, k, n, m;
    float *priors;
    for (n=0,m=0; n<4; n++)
        for (j=0; j<featuremap_size[1][n]; j++)
            for (i=0; i<featuremap_size[0][n]; i++)
                for (k = 0; k < 3 && MINBOXES[n][k] != 0; k++, m++);
    if (num_priors) *num_priors = m;
    priors = malloc(m * 4 * sizeof(float));
    if (!priors) return NULL;

    for (n=0,m=0; n<4; n++) {
        float scalew = inputw / SHRINKAGES[0][n];
        float scaleh = inputh / SHRINKAGES[1][n];
        for (j=0; j<featuremap_size[1][n]; j++) {
            for (i=0; i<featuremap_size[0][n]; i++) {
                float x_center = (i + 0.5) / scalew;
                float y_center = (j + 0.5) / scaleh;
                for (k = 0; k < 3 && MINBOXES[n][k] != 0; k++, m++) {
                    float w = MINBOXES[n][k] / inputw;
                    float h = MINBOXES[n][k] / inputh;
                    priors[m * 4 + 0] = CLIP(x_center, 1);
                    priors[m * 4 + 1] = CLIP(y_center, 1);
                    priors[m * 4 + 2] = CLIP(w       , 1);
                    priors[m * 4 + 3] = CLIP(h       , 1);
                }
            }
        }
    }
    return priors;
}

static int ufface_postprocess_proc(int inputw, int inputh, float *priors_list, int priors_num, float *tensor_scores, float *tensor_boxes, BBOX *bboxlist, int bboxmaxnum)
{
    int  i, n;
    for (i=0,n=0; i<priors_num; i++) {
        if (tensor_scores[i * 2 + 1] > SCORE_THRESH) {
            static const float CENTER_VARIANCE = 0.1;
            static const float SIZE_VARIANCE   = 0.2;
            float x_center = tensor_boxes[i * 4 + 0] * CENTER_VARIANCE * priors_list[i * 4 + 2] + priors_list[i * 4 + 0];
            float y_center = tensor_boxes[i * 4 + 1] * CENTER_VARIANCE * priors_list[i * 4 + 3] + priors_list[i * 4 + 1];
            float w = exp(tensor_boxes[i * 4 + 2] * SIZE_VARIANCE) * priors_list[i * 4 + 2];
            float h = exp(tensor_boxes[i * 4 + 3] * SIZE_VARIANCE) * priors_list[i * 4 + 3];
            if (n < bboxmaxnum) {
                bboxlist[n].type  = 0;
                bboxlist[n].score = tensor_scores[i * 2 + 1];
                bboxlist[n].x1    = CLIP(x_center - w / 2.0, 1) * inputw;
                bboxlist[n].y1    = CLIP(y_center - h / 2.0, 1) * inputh;
                bboxlist[n].x2    = CLIP(x_center + w / 2.0, 1) * inputw;
                bboxlist[n].y2    = CLIP(y_center + h / 2.0, 1) * inputh;
                n++;
            }
        }
    }
    return n;
}

int main(int argc, char *argv[])
{
    char  *model_type    = "yolo-fastest";
    char  *sgs_model_out = "out.txt";
    int    picture_width = MODEL_INPUT_WIDTH ;
    int    picture_height= MODEL_INPUT_HEIGHT;
    BBOX   bbox_list[MAX_BBOX_NUM];
    int    bbox_cursize = 0, i;
    int    yolofast_classnum  = MODEL_CLASS_NUMBER;
    float *ufface_priors_list = NULL;
    int    ufface_priors_num  = 0;

    if (argc > 1) sgs_model_out     = argv[1];
    if (argc > 2) picture_width     = atoi(argv[2]);
    if (argc > 3) picture_height    = atoi(argv[3]);
    if (argc > 4) model_type        = argv[4];
    if (argc > 5) yolofast_classnum = atoi(argv[5]);
    printf("sgs_model_out : %s\n", sgs_model_out );
    printf("picture_width : %d\n", picture_width );
    printf("picture_height: %d\n", picture_height);
    printf("model_type    : %s\n", model_type    );

    if (strcmp(model_type, "yolo-fastest") == 0) {
        for (i=0; i<MODEL_OUTPUT_NUM; i++) {
            float *data = NULL; int h = 0, w = 0, c = 0;
            if (0 != load_output_data(sgs_model_out, i, &data, &h, &w, &c)) break;
            bbox_cursize = yolov3(data, h, w, c, yolofast_classnum, SCORE_THRESH, s_anchor_list[i], bbox_list, bbox_cursize, MAX_BBOX_NUM);
            free_output_data(data);
        }

        bbox_cursize = nms(bbox_list, bbox_cursize, NMSIOU_THRESH, 1, picture_width, picture_height);
        for (i=0; i<bbox_cursize; i++) {
            printf("score: %.2f, category: %2d, rect: (%3d %3d %3d %3d)\n", bbox_list[i].score, bbox_list[i].type,
                (int)bbox_list[i].x1, (int)bbox_list[i].y1, (int)bbox_list[i].x2, (int)bbox_list[i].y2);
        }
    } else if (strcmp(model_type, "ultra-facedet") == 0) {
        float *data[MODEL_OUTPUT_NUM] = { NULL };
        ufface_priors_list = ufface_init_priors(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, &ufface_priors_num);
        if (!ufface_priors_list) {
            printf("failed to init ultra-facedet priors !\n");
            return -1;
        }

        for (i=0; i<MODEL_OUTPUT_NUM; i++) {
            int h = 0, w = 0, c = 0;
            if (0 != load_output_data(sgs_model_out, i, &(data[i]), &h, &w, &c)) {
                printf("failed to load output sensor %d !\n", i);
                break;
            }
        }
        if (MODEL_OUTPUT_NUM >= 2 && data[0] && data[1]) {
            bbox_cursize = ufface_postprocess_proc(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, ufface_priors_list, ufface_priors_num, data[0], data[1], bbox_list, MAX_BBOX_NUM);
            bbox_cursize = nms(bbox_list, bbox_cursize, NMSIOU_THRESH, 1, picture_width, picture_height);
        }
        for (i=0; i<MODEL_OUTPUT_NUM; i++) free_output_data(data[i]);
        for (i=0; i<bbox_cursize; i++) {
            printf("score: %.2f, category: %2d, rect: (%3d %3d %3d %3d)\n", bbox_list[i].score, bbox_list[i].type,
                (int)bbox_list[i].x1, (int)bbox_list[i].y1, (int)bbox_list[i].x2, (int)bbox_list[i].y2);
        }
        free(ufface_priors_list);
    }

    return 0;
}
