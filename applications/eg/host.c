
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
int main(int argc, char *argv[])
{

  char *hostname;
  struct hostent *myhost;
  char **aliaslist;
  char **addresslist;
  int i = 0;
  struct in_addr myaddr;
  long int tln_addr;
  char *dot_addr;

  if (argc < 2) {
    printf("Useage is $ ./host_info [host name] \n");
    exit(-1);
  }
  else {
    hostname = (char *)malloc(20);
    strcpy(hostname, argv[1]);
  }
  
  myhost = gethostbyname(hostname);

  aliaslist = myhost->h_aliases;
  addresslist = myhost->h_addr_list;
  printf("Official host name: %s \n", myhost->h_name);
  printf("Host address type is: %d\n", myhost->h_addrtype);
  printf("Length of address is : %d\n", myhost->h_length);

  if (*aliaslist) { 
   
       while (*aliaslist != NULL) {
         printf("%s \n", *aliaslist);
         *aliaslist++;
       }
  }
  else 
       printf("No aliases \n");

  if (*addresslist) {
    while (*addresslist != NULL) {
        myaddr.s_addr = **addresslist;
        dot_addr = inet_ntoa(myaddr);
        printf("%s \n", dot_addr);
        *addresslist++;
    }
  }
  else
    printf("No addresses... \n");


 free(hostname);  

 return 0;




}

