#ifndef ERROR_H
#define ERROR_H

// Text formatting macros
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[92;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
#define BOLD(x) "\033[1m" x "\033[0m"

#define FATAL_ERROR(msg, ...) \
    do { \
        fprintf(stderr, RED_TEXT("FATAL [%s:%d in %s()]: " msg "\n"), \
                __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
        exit(1); \
    } while(0)

#define ERROR(msg, ...) \
    fprintf(stderr, RED_TEXT("ERROR [%s:%d in %s()]: " msg "\n"), \
            __FILE__, __LINE__, __func__, ##__VA_ARGS__)

#define WARNING(msg, ...) \
    fprintf(stderr, YELLOW_TEXT("WARNING [%s:%d in %s()]: " msg "\n"), \
            __FILE__, __LINE__, __func__, ##__VA_ARGS__)            
#endif 