#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/wait.h>

#define BUFF_SIZE 1024

void activateScreensaver() {
    system("xdg-screensaver activate");
}


void openPDF(const char* filename) {
    char command[256];
    snprintf(command, sizeof(command), "xdg-open %s", filename);
    system(command);
}

int main(void){
    int server_socket;
    int client_socket;
    int client_addr_size;

    struct sockaddr_in server_addr;
    struct sockaddr_in client_addr;

    char buff_rcv[BUFF_SIZE + 5];
    char buff_snd[BUFF_SIZE + 5];

    
    server_socket = socket(PF_INET, SOCK_STREAM, 0);
    if (-1 == server_socket){
        printf("server socket 생성 실패\n");
        exit(1);
    }
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(4000);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    //printf(INADDR_ANY);
    printf("dg's hope\n");

    if (-1 == bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr))){
        // printf( "bind() 실행 에러\n");
        // exit( 1);
    }
    if (-1 == listen(server_socket, 5)){
        printf("listen() 실행 실패\n");
        exit(1);
    }
    int Buser=2;
    while (1){
        client_addr_size = sizeof(client_addr);
        client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_addr_size);

        if (-1 == client_socket){
            printf("클라이언트 연결 수락 실패\n");
            exit(1);
        }
        read(client_socket, buff_rcv, BUFF_SIZE);
        if(buff_rcv[0]=='0'){
            printf("사용자:조수호\n");
            if(Buser!=0) {
            activateScreensaver();
            sleep(5);
            }
            Buser=0;
        }
        else if(buff_rcv[0]=='1'){
            printf("사용자:김예성\n");
            if(Buser!=1) {
                const char* filename = "9주차수업자료_증기동력시스템.pdf";
    	        openPDF(filename);
    	        sleep(5);
            }
            Buser=1;
        }
        else if(buff_rcv[0]=='2'){
            printf("사용자:배재형\n");//
            
            if(Buser!=2) {
            activateScreensaver();
            sleep(5);
            }
            Buser=2;
        }//
        else if(buff_rcv[0]=='3'){
            printf("사용자:None\n");
            if(Buser!=3){
            pid_t child_pid = fork();
    	    if (child_pid == -1) {
        	perror("Failed to fork");
        	exit(1);
    	   } else if (child_pid == 0) {
           // 자식 프로세스에서 다른 사용자의 데스크톱 세션으로 전환
           execl("/usr/bin/gdmflexiserver", "gdmflexiserver", "—xnest", NULL);
           perror("Failed to execute gdmflexiserver");
           exit(1);
    	   } else {
           // 부모 프로세스에서 자식 프로세스가 종료될 때까지 대기
           int status;
           waitpid(child_pid, &status, 0);
    	   }
    	   sleep(5);
    	   }
    	   Buser=3;
        }
        printf("receive: %s\n", buff_rcv);

        // snprintf( buff_snd, "%zu : %s", strlen( buff_rcv), buff_rcv);
        // snprintf(buff_snd, sizeof(buff_snd), "%zu : %s", strlen(buff_rcv), buff_rcv);
        snprintf(buff_snd, sizeof(buff_snd), "%zu : %.1000s", strlen(buff_rcv), buff_rcv);
        write(client_socket, buff_snd, strlen(buff_snd) + 1); // +1: NULL까지 포함해서 전송
        close(client_socket);
    }
}
