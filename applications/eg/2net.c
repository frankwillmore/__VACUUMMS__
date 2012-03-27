//#include <netdb.h>
//#include <sys/socket.h>
//#include <sys/param.h>
//#include <sys/types.h>
//#include <netinet/in.h>
#include <arpa/inet.h>
//#include <unistd.h>


char      name_buf[25] = "";
struct hostent *hp = NULL;


main()
{
 // hp = gethostbyname(name_buf);
 gethostname(name_buf, (25 - 1));

  printf("%s\n", name_buf);

}
