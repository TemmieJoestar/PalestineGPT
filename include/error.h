#ifndef ERROR_H
#define ERROR_H

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