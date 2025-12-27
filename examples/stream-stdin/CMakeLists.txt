set(TARGET whisper-stream-stdin)
add_executable(${TARGET} stream-stdin.cpp)
target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})
