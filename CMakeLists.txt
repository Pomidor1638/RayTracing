cmake_minimum_required(VERSION 3.20)
project(RayTracing)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp

    src/imgui/imconfig.h
    src/imgui/imgui.h
    
    
    src/imgui/imgui_impl_sdl2.h
    src/imgui/imgui_impl_sdlrenderer2.h
    src/imgui/imgui_internal.h
    src/imgui/imstb_rectpack.h
    src/imgui/imstb_textedit.h
    src/imgui/imstb_truetype.h

    src/imgui/imgui.cpp
    src/imgui/imgui_draw.cpp
    
    src/imgui/imgui_impl_sdl2.cpp
    src/imgui/imgui_impl_sdlrenderer2.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui_widgets.cpp

)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    SDL2::SDL2main
    SDL2::SDL2
    glm::glm
)