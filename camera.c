#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#define VIDEO_DEVICE "/dev/video0"
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define IMAGE_FILE "test_data/test_data.jpg"

int main() {
    // 카메라 장치 열기
    int video_fd = open(VIDEO_DEVICE, O_RDWR);
    if (video_fd == -1) {
        perror("카메라 장치를 열 수 없습니다.");
        return 1;
    }

    // 영상 포맷 설정
    struct v4l2_format format;
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = IMAGE_WIDTH;
    format.fmt.pix.height = IMAGE_HEIGHT;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(video_fd, VIDIOC_S_FMT, &format) == -1) {
        perror("영상 포맷 설정에 실패했습니다.");
        close(video_fd);
        return 1;
    }

    // 버퍼 메모리 할당
    struct v4l2_requestbuffers reqbuf;
    reqbuf.count = 1;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(video_fd, VIDIOC_REQBUFS, &reqbuf) == -1) {
        perror("버퍼 메모리 할당에 실패했습니다.");
        close(video_fd);
        return 1;
    }

    struct v4l2_buffer buffer;
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = 0;
    if (ioctl(video_fd, VIDIOC_QUERYBUF, &buffer) == -1) {
        perror("버퍼 정보를 가져올 수 없습니다.");
        close(video_fd);
        return 1;
    }

    void* buffer_start = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, video_fd, buffer.m.offset);
    if (buffer_start == MAP_FAILED) {
        perror("버퍼 메모리를 매핑할 수 없습니다.");
        close(video_fd);
        return 1;
    }

    // 영상 캡처 시작
    if (ioctl(video_fd, VIDIOC_STREAMON, &buffer.type) == -1) {
        perror("영상 스트리밍을 시작할 수 없습니다.");
        munmap(buffer_start, buffer.length);
        close(video_fd);
        return 1;
    }
    while (1){
        // 영상 데이터 읽기
        if (ioctl(video_fd, VIDIOC_QBUF, &buffer) == -1) {
            perror("버퍼를 큐에 넣을 수 없습니다.");
            munmap(buffer_start, buffer.length);
            close(video_fd);
            return 1;
        }

        if (ioctl(video_fd, VIDIOC_DQBUF, &buffer) == -1) {
            perror("버퍼에서 데이터를 읽을 수 없습니다.");
            munmap(buffer_start, buffer.length);
            close(video_fd);
            return 1;
        }

        // 영상 데이터 저장
        char* image_path = IMAGE_FILE;
        FILE* image_file = fopen(image_path, "wb");
        if (image_file == NULL) {
            perror("이미지 파일을 열 수 없습니다.");
            munmap(buffer_start, buffer.length);
            close(video_fd);
            return 1;
        }

        fwrite(buffer_start, buffer.length, 1, image_file);
        fclose(image_file);

        sleep(1.7)
    }

    // 영상 캡처 종료
    if (ioctl(video_fd, VIDIOC_STREAMOFF, &buffer.type) == -1) {
        perror("영상 스트리밍을 종료할 수 없습니다.");
        munmap(buffer_start, buffer.length);
        close(video_fd);
        return 1;
    }

    // 메모리 해제 및 카메라 장치 닫기
    munmap(buffer_start, buffer.length);
    close(video_fd);

    printf("이미지가 성공적으로 저장되었습니다.\n");

    return 0;
}
