#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MODEL_INPUT_WIDTH   320
#define MODEL_INPUT_HEIGHT  320
#define MODEL_CLASS_NUMBER  80
#define MODEL_OUTPUT_NUM    2
#define MAX_BBOX_NUM        200
#define SCORE_THRESH        0.5
#define NMSIOU_THRESH       0.5

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
    int   fsize, ok = 0, tmp, i;

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
        pcur = strstr(pstart, "lignment shape:[");
        if (!pcur) pcur = strstr(pstart, "Original shape:[");
        if (!pcur) {
            printf("failed to read data %d !\n", idx);
            goto done;
        }
        pstart = pcur + strlen("Original shape:[");
    }
    
    pcur += strlen("Original shape:[");
    sscanf(pcur, "%d %d %d %d", &tmp, h, w, c);
    if (*h <= 0 || *w <= 0 || *c <= 0) {
        printf("invalid h/w/c value ! h: %d, w: %d, c: %d\n", *h, *w, *c);
        goto done;
    } else {
        printf("output sensor idx: %d, h: %d, w: %d, c: %d\n", idx, *h, *w, *c);
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

int main(int argc, char *argv[])
{
    char *sgs_model_out = "out.txt";
    int   picture_width = MODEL_INPUT_WIDTH ;
    int   picture_height= MODEL_INPUT_HEIGHT;
    BBOX  bbox_list[MAX_BBOX_NUM];
    int   bbox_cursize = 0, i;
    
    if (argc > 1) sgs_model_out = argv[1];
    if (argc > 2) picture_width = atoi(argv[2]);
    if (argc > 3) picture_height= atoi(argv[3]);
    printf("sgs_model_out : %s\n", sgs_model_out );
    printf("picture_width : %d\n", picture_width );
    printf("picture_height: %d\n", picture_height);
    
    for (i=0; i<MODEL_OUTPUT_NUM; i++) {
        float *data = NULL; int h = 0, w = 0, c = 0;
        if (0 != load_output_data(sgs_model_out, i, &data, &h, &w, &c)) break;
        bbox_cursize = yolov3(data, h, w, c, MODEL_CLASS_NUMBER, SCORE_THRESH, s_anchor_list[i], bbox_list, bbox_cursize, MAX_BBOX_NUM);
        free_output_data(data);
    }

    bbox_cursize = nms(bbox_list, bbox_cursize, NMSIOU_THRESH, 1, picture_width, picture_height);
    for (i=0; i<bbox_cursize; i++) {
        printf("score: %.2f, category: %2d, rect: (%3d %3d %3d %3d)\n", bbox_list[i].score, bbox_list[i].type,
            (int)bbox_list[i].x1, (int)bbox_list[i].y1, (int)bbox_list[i].x2, (int)bbox_list[i].y2);
    }

    return 0;
}
