#include <SDL2/SDL.h>
#include <thread>
#include <atomic>
#include <vector>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <array>
#include <cmath>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl2.h"
#include "imgui/imgui_impl_sdlrenderer2.h"

const int DEFAULT_WIDTH = 1024;
const int DEFAULT_HEIGHT = 768;
const int NUM_THREADS = std::max(1, (int)std::thread::hardware_concurrency());
const int MAX_BOUNCES = 5;
const float DEFAULT_ATTENUATION = 1.0f;
const int TARGET_FPS = 30;

using Vec3 = glm::vec3;
using Point3 = glm::vec3;
using Color = glm::vec3;

struct Ray {
    Point3 origin;
    Vec3 direction;

    Ray() = default;
    Ray(const Point3& origin, const Vec3& direction)
        : origin(origin), direction(direction) {
    }

    Point3 At(float t) const {
        return origin + direction * t;
    }
};

struct Camera {
    Point3 position;
    float yaw;
    float pitch;
    float fov;

    Camera(const Point3& pos = Point3(0, 0, 0),
        float yaw = 0.0f,
        float pitch = 0.0f,
        float fov = 1.5f)
        : position(pos), yaw(yaw), pitch(pitch), fov(fov) {
    }

    struct CameraBasis {
        Vec3 forward;
        Vec3 right;
        Vec3 up;
        float fov;
    };

    CameraBasis GetBasis() const {
        CameraBasis basis;

        float yaw_rad = glm::radians(yaw);
        float pitch_rad = glm::radians(pitch);

        float cos_yaw = std::cos(yaw_rad);
        float sin_yaw = std::sin(yaw_rad);
        float cos_pitch = std::cos(pitch_rad);
        float sin_pitch = std::sin(pitch_rad);

        basis.forward = glm::normalize(Vec3(cos_yaw * cos_pitch, sin_pitch, -sin_yaw * cos_pitch));
        basis.right = glm::normalize(glm::cross(basis.forward, Vec3(0, 1, 0)));
        basis.up = glm::cross(basis.right, basis.forward);
        basis.fov = fov;

        return basis;
    }

    Ray GetRayForPixel(const CameraBasis& basis, float screen_x, float screen_y) const {
        Vec3 direction = glm::normalize(basis.forward * basis.fov + screen_x * basis.right + screen_y * basis.up);
        return Ray(position, direction);
    }
};

struct Input {
    bool w = false;
    bool a = false;
    bool s = false;
    bool d = false;
    bool shift = false;
    bool gui_visible = true;
    bool mouse_captured = false;
    float mouse_sensitivity = 0.1f;
    float move_speed = 5.0f;
};

struct HitRecord {
    Point3 position;
    Vec3 normal;
    float t;
    bool front_face;
    Color color;
    float reflectivity;
    float emission;

    void SetFaceNormal(const Ray& ray, const Vec3& outward_normal) {
        front_face = glm::dot(ray.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct Entity {
    Point3 center;
    Color color;
    float reflectivity;
    std::string name;
    float emission;
    bool solid = true;

    float rotation_yaw = 0.0f;
    float rotation_pitch = 0.0f;
    float rotation_roll = 0.0f;

    glm::mat4 rotation_matrix;
    glm::mat4 inverse_rotation;

    Entity(const Point3& center, const Color& color, float reflectivity, const std::string& name, float emission = 0.0f)
        : center(center), color(color), reflectivity(reflectivity), name(name), emission(emission) {
        UpdateRotationMatrix();
    }

    virtual ~Entity() = default;

    void UpdateRotationMatrix() {
        float yaw_rad = glm::radians(rotation_yaw);
        float pitch_rad = glm::radians(rotation_pitch);
        float roll_rad = glm::radians(rotation_roll);

        glm::mat4 rot_y = glm::rotate(glm::mat4(1.0f), yaw_rad, Vec3(0, 1, 0));
        glm::mat4 rot_x = glm::rotate(glm::mat4(1.0f), pitch_rad, Vec3(1, 0, 0));
        glm::mat4 rot_z = glm::rotate(glm::mat4(1.0f), roll_rad, Vec3(0, 0, 1));

        rotation_matrix = rot_y * rot_x * rot_z;
        inverse_rotation = glm::transpose(rotation_matrix);
    }

    virtual bool Hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const = 0;
    virtual Vec3 GetNormal(const Point3& hit_point) const = 0;
};

struct Light : public Entity {
    float radius;
    float radius_squared;
    float intensity;

    Light(const Point3& center, float radius, const Color& color, float intensity = 1.0f)
        : Entity(center, color, 0.0f, "Light", 1.0f),
        radius(radius), radius_squared(radius* radius), intensity(intensity) {
        solid = false;
    }

    bool Hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const override {
        Vec3 oc = ray.origin - center;
        float a = glm::dot(ray.direction, ray.direction);
        float b = glm::dot(oc, ray.direction);
        float c = glm::dot(oc, oc) - radius_squared;

        float discriminant = b * b - a * c;
        if (discriminant < 0) return false;

        float sqrt_disc = std::sqrt(discriminant);
        float t = (-b - sqrt_disc) / a;
        if (t < t_min || t > t_max) {
            t = (-b + sqrt_disc) / a;
            if (t < t_min || t > t_max) return false;
        }

        record.t = t;
        record.position = ray.At(t);
        record.color = color * intensity;
        record.reflectivity = 0.0f;
        record.emission = 1.0f;

        Vec3 outward_normal = (record.position - center) / radius;
        record.SetFaceNormal(ray, outward_normal);

        if (solid && !record.front_face) return false;

        return true;
    }

    Vec3 GetNormal(const Point3& hit_point) const override {
        return glm::normalize(hit_point - center);
    }
};

struct Sphere : public Entity {
    float radius;
    float radius_squared;

    Sphere(const Point3& center, float radius, const Color& color, float reflectivity = 0.0f)
        : Entity(center, color, reflectivity, "Sphere"),
        radius(radius), radius_squared(radius* radius) {
        solid = true;
    }

    bool Hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const override {
        Vec3 oc = ray.origin - center;
        float a = glm::dot(ray.direction, ray.direction);
        float b = glm::dot(oc, ray.direction);
        float c = glm::dot(oc, oc) - radius_squared;

        float discriminant = b * b - a * c;
        if (discriminant < 0) return false;

        float sqrt_disc = std::sqrt(discriminant);
        float t = (-b - sqrt_disc) / a;
        if (t < t_min || t > t_max) {
            t = (-b + sqrt_disc) / a;
            if (t < t_min || t > t_max) return false;
        }

        record.t = t;
        record.position = ray.At(t);
        record.color = color;
        record.reflectivity = reflectivity;
        record.emission = emission;

        Vec3 outward_normal = (record.position - center) / radius;
        record.SetFaceNormal(ray, outward_normal);

        if (solid && !record.front_face) return false;

        return true;
    }

    Vec3 GetNormal(const Point3& hit_point) const override {
        return glm::normalize(hit_point - center);
    }
};

struct Ellipsoid : public Entity {
    Vec3 radii;
    Vec3 radii_squared;

    Ellipsoid(const Point3& center, const Vec3& radii, const Color& color, float reflectivity = 0.0f)
        : Entity(center, color, reflectivity, "Ellipsoid"),
        radii(radii), radii_squared(radii.x* radii.x, radii.y* radii.y, radii.z* radii.z) {
        solid = true;
    }

    bool Hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const override {
        Point3 local_origin = ray.origin - center;

        Vec3 local_origin_rotated = glm::vec3(inverse_rotation * glm::vec4(local_origin, 0.0f));
        Vec3 local_dir_rotated = glm::vec3(inverse_rotation * glm::vec4(ray.direction, 0.0f));

        Vec3 scaled_origin(local_origin_rotated.x / radii.x, local_origin_rotated.y / radii.y, local_origin_rotated.z / radii.z);
        Vec3 scaled_dir(local_dir_rotated.x / radii.x, local_dir_rotated.y / radii.y, local_dir_rotated.z / radii.z);

        float a = glm::dot(scaled_dir, scaled_dir);
        float b = glm::dot(scaled_origin, scaled_dir);
        float c = glm::dot(scaled_origin, scaled_origin) - 1.0f;

        float discriminant = b * b - a * c;
        if (discriminant < 0) return false;

        float sqrt_disc = std::sqrt(discriminant);
        float t = (-b - sqrt_disc) / a;
        if (t < t_min || t > t_max) {
            t = (-b + sqrt_disc) / a;
            if (t < t_min || t > t_max) return false;
        }

        record.t = t;
        record.position = ray.At(t);
        record.color = color;
        record.reflectivity = reflectivity;
        record.emission = emission;

        Point3 local_hit = local_origin_rotated + local_dir_rotated * t;
        Vec3 gradient(local_hit.x / radii_squared.x, local_hit.y / radii_squared.y, local_hit.z / radii_squared.z);
        Vec3 local_normal = glm::normalize(gradient);
        Vec3 world_normal = glm::vec3(rotation_matrix * glm::vec4(local_normal, 0.0f));
        record.SetFaceNormal(ray, world_normal);

        if (solid && !record.front_face) return false;

        return true;
    }

    Vec3 GetNormal(const Point3& hit_point) const override {
        Point3 local_hit = hit_point - center;
        Vec3 local_hit_rotated = glm::vec3(inverse_rotation * glm::vec4(local_hit, 0.0f));
        Vec3 gradient(local_hit_rotated.x / radii_squared.x, local_hit_rotated.y / radii_squared.y, local_hit_rotated.z / radii_squared.z);
        Vec3 local_normal = glm::normalize(gradient);
        return glm::normalize(glm::vec3(rotation_matrix * glm::vec4(local_normal, 0.0f)));
    }
};

struct Box : public Entity {
    Vec3 size;

    Box(const Point3& center, const Vec3& size, const Color& color, float reflectivity = 0.0f)
        : Entity(center, color, reflectivity, "Box"), size(size) {
        solid = true;
    }

    bool Hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const override {
        Point3 local_origin = ray.origin - center;

        Vec3 local_origin_rotated = glm::vec3(inverse_rotation * glm::vec4(local_origin, 0.0f));
        Vec3 local_dir_rotated = glm::vec3(inverse_rotation * glm::vec4(ray.direction, 0.0f));

        Vec3 inv_dir(
            local_dir_rotated.x != 0 ? 1.0f / local_dir_rotated.x : 1e10f,
            local_dir_rotated.y != 0 ? 1.0f / local_dir_rotated.y : 1e10f,
            local_dir_rotated.z != 0 ? 1.0f / local_dir_rotated.z : 1e10f
        );

        float t1 = (-size.x - local_origin_rotated.x) * inv_dir.x;
        float t2 = (size.x - local_origin_rotated.x) * inv_dir.x;
        float t3 = (-size.y - local_origin_rotated.y) * inv_dir.y;
        float t4 = (size.y - local_origin_rotated.y) * inv_dir.y;
        float t5 = (-size.z - local_origin_rotated.z) * inv_dir.z;
        float t6 = (size.z - local_origin_rotated.z) * inv_dir.z;

        float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

        if (tmax < 0 || tmin > tmax) return false;

        float t = tmin < t_min ? tmax : tmin;
        if (t < t_min || t > t_max) return false;

        record.t = t;
        record.position = ray.At(t);
        record.color = color;
        record.reflectivity = reflectivity;
        record.emission = emission;

        Point3 local_hit = local_origin_rotated + local_dir_rotated * t;
        Vec3 local_normal(0, 0, 0);
        float epsilon = size.x * 0.01f;

        if (std::abs(local_hit.x - size.x) < epsilon) local_normal = Vec3(1, 0, 0);
        else if (std::abs(local_hit.x + size.x) < epsilon) local_normal = Vec3(-1, 0, 0);
        else if (std::abs(local_hit.y - size.y) < epsilon) local_normal = Vec3(0, 1, 0);
        else if (std::abs(local_hit.y + size.y) < epsilon) local_normal = Vec3(0, -1, 0);
        else if (std::abs(local_hit.z - size.z) < epsilon) local_normal = Vec3(0, 0, 1);
        else if (std::abs(local_hit.z + size.z) < epsilon) local_normal = Vec3(0, 0, -1);

        Vec3 world_normal = glm::vec3(rotation_matrix * glm::vec4(local_normal, 0.0f));
        record.SetFaceNormal(ray, world_normal);

        if (solid && !record.front_face) return false;

        return true;
    }

    Vec3 GetNormal(const Point3& hit_point) const override {
        Point3 local_hit = hit_point - center;
        Vec3 local_hit_rotated = glm::vec3(inverse_rotation * glm::vec4(local_hit, 0.0f));

        Vec3 local_normal(0, 0, 0);
        float epsilon = size.x * 0.01f;

        if (std::abs(local_hit_rotated.x - size.x) < epsilon) local_normal = Vec3(1, 0, 0);
        else if (std::abs(local_hit_rotated.x + size.x) < epsilon) local_normal = Vec3(-1, 0, 0);
        else if (std::abs(local_hit_rotated.y - size.y) < epsilon) local_normal = Vec3(0, 1, 0);
        else if (std::abs(local_hit_rotated.y + size.y) < epsilon) local_normal = Vec3(0, -1, 0);
        else if (std::abs(local_hit_rotated.z - size.z) < epsilon) local_normal = Vec3(0, 0, 1);
        else if (std::abs(local_hit_rotated.z + size.z) < epsilon) local_normal = Vec3(0, 0, -1);

        return glm::normalize(glm::vec3(rotation_matrix * glm::vec4(local_normal, 0.0f)));
    }
};

struct Floor {
    float y;
    Color color1;
    Color color2;
    float checkSize;

    Floor(float y, const Color& color1, const Color& color2, float checkSize = 1.0f)
        : y(y), color1(color1), color2(color2), checkSize(checkSize) {
    }

    bool Hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const {
        if (std::abs(ray.direction.y) < 0.0001f) return false;

        float t = (y - ray.origin.y) / ray.direction.y;
        if (t < t_min || t > t_max) return false;

        record.t = t;
        record.position = ray.At(t);
        record.normal = Vec3(0, 1, 0);
        record.front_face = ray.direction.y < 0;
        record.reflectivity = 0.1f;
        record.emission = 0.0f;

        if (!record.front_face) return false;

        float x = record.position.x / checkSize;
        float z = record.position.z / checkSize;
        int check = (int(std::floor(x)) + int(std::floor(z))) % 2;
        record.color = check == 0 ? color1 : color2;

        return true;
    }
};

struct World {
    std::vector<std::shared_ptr<Entity>> entities;
    std::vector<std::shared_ptr<Light>> lights;
    Floor floor;

    float ambient_strength;

    World(const Floor& floor) : floor(floor) {
        ambient_strength = 0.05f;
    }

    void AddEntity(std::shared_ptr<Entity> entity) {
        entities.push_back(entity);
    }

    void AddLight(std::shared_ptr<Light> light) {
        lights.push_back(light);
        entities.push_back(light);
    }

    bool Hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const {
        HitRecord temp_record;
        bool hit_anything = false;
        auto closest_so_far = t_max;

        for (const auto& entity : entities) {
            if (entity->Hit(ray, t_min, closest_so_far, temp_record)) {
                hit_anything = true;
                closest_so_far = temp_record.t;
                record = temp_record;
            }
        }

        if (floor.Hit(ray, t_min, closest_so_far, temp_record)) {
            hit_anything = true;
            closest_so_far = temp_record.t;
            record = temp_record;
        }

        return hit_anything;
    }
};

SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
SDL_Texture* texture = nullptr;

std::atomic<bool> g_running{ true };

std::array<std::vector<Uint32>, 2> g_pixels;
std::atomic<int> g_current_buffer{ 0 };
std::atomic<bool> g_frame_ready{ false };

std::atomic<int> g_render_width{ DEFAULT_WIDTH };
std::atomic<int> g_render_height{ DEFAULT_HEIGHT };

int g_config_width = DEFAULT_WIDTH;
int g_config_height = DEFAULT_HEIGHT;

const int MAX_WIDTH = 1920;
const int MAX_HEIGHT = 1080;

std::mutex g_mutex;
std::condition_variable g_cv;
std::atomic<int> g_threads_done{ 0 };

World* g_world = nullptr;
Camera* g_camera = nullptr;
Input g_input;
float g_attenuation = DEFAULT_ATTENUATION;

Camera::CameraBasis g_camera_basis;

int g_selected_entity = -1;
std::vector<std::string> g_entity_names;

void ProcessInput(float delta_time) {
    if (!g_input.gui_visible) {
        float speed = g_input.shift ? g_input.move_speed * 3.0f : g_input.move_speed;

        float yaw_rad = glm::radians(g_camera->yaw);
        float pitch_rad = glm::radians(g_camera->pitch);

        Vec3 forward(std::cos(yaw_rad) * std::cos(pitch_rad),
            std::sin(pitch_rad),
            -std::sin(yaw_rad) * std::cos(pitch_rad));
        Vec3 right(std::sin(yaw_rad), 0, std::cos(yaw_rad));

        if (g_input.w) g_camera->position += forward * speed * delta_time;
        if (g_input.s) g_camera->position -= forward * speed * delta_time;
        if (g_input.a) g_camera->position -= right * speed * delta_time;
        if (g_input.d) g_camera->position += right * speed * delta_time;

        g_camera->position.y = glm::clamp(g_camera->position.y, -5.0f, 10.0f);
    }
}

void HandleMouseEvent(const SDL_MouseMotionEvent& event) {
    if (!g_input.gui_visible && g_input.mouse_captured) {
        g_camera->yaw -= event.xrel * g_input.mouse_sensitivity;
        g_camera->pitch -= event.yrel * g_input.mouse_sensitivity;

        g_camera->pitch = glm::clamp(g_camera->pitch, -89.0f, 89.0f);

        if (g_camera->yaw > 180.0f) g_camera->yaw -= 360.0f;
        if (g_camera->yaw < -180.0f) g_camera->yaw += 360.0f;
    }
}

void ToggleMouseCapture() {
    g_input.mouse_captured = !g_input.mouse_captured;
    SDL_SetRelativeMouseMode(g_input.mouse_captured ? SDL_TRUE : SDL_FALSE);
}

void RecreateTexture() {
    int new_width = g_render_width.load();
    int new_height = g_render_height.load();

    if (texture) SDL_DestroyTexture(texture);

    texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        new_width, new_height);

    if (!texture) {
        SDL_Log("Failed to create texture: %s", SDL_GetError());
        return;
    }

    SDL_Log("Resolution changed to %dx%d", new_width, new_height);
}

Color CalculateLighting(const HitRecord& record, const World& world, const Ray& view_ray) {
    Color final_color = record.color * world.ambient_strength;

    for (const auto& light : world.lights) {
        Vec3 light_dir = glm::normalize(light->center - record.position);
        float light_dist = glm::length(light->center - record.position);
        float attenuation = 1.0f / (1.0f + 0.1f * light_dist + 0.01f * light_dist * light_dist);

        float diff = glm::max(glm::dot(record.normal, light_dir), 0.0f);
        Color diffuse = record.color * light->color * diff * light->intensity * attenuation;

        Ray shadow_ray(record.position + record.normal * 0.01f, light_dir);
        HitRecord shadow_record;
        bool in_shadow = false;

        for (const auto& entity : world.entities) {
            if (entity.get() == light.get()) continue;
            if (entity->Hit(shadow_ray, 0.01f, light_dist - 0.1f, shadow_record)) {
                in_shadow = true;
                break;
            }
        }

        if (!in_shadow) {
            final_color += diffuse;
        }

        if (record.reflectivity > 0.0f) {
            Vec3 reflect_dir = glm::reflect(view_ray.direction, record.normal);
            final_color = glm::mix(final_color, record.color * attenuation, record.reflectivity);
        }
    }

    if (record.emission > 0.0f) {
        final_color += record.color * 2.0f;
    }

    return final_color;
}

Color TraceRay(const Ray& ray, const World& world, int bounce_count) {
    HitRecord record;

    if (!world.Hit(ray, 0.001f, 10000.0f, record)) {
        return Color(0.02f, 0.02f, 0.05f);
    }

    if (record.emission > 0.0f) {
        return record.color * 2.0f;
    }

    Color lighting = CalculateLighting(record, world, ray);

    if (record.reflectivity > 0.0f && bounce_count < MAX_BOUNCES) {
        Vec3 reflect_dir = glm::reflect(ray.direction, record.normal);
        Ray reflect_ray(record.position + record.normal * 0.001f, reflect_dir);
        Color reflect_color = TraceRay(reflect_ray, world, bounce_count + 1);
        lighting = glm::mix(lighting, reflect_color * 0.5f, record.reflectivity);
    }

    float attenuation = std::pow(g_attenuation, float(bounce_count));
    return lighting * attenuation;
}

Uint32 ColorToPixel(const Color& color) {
    Uint8 r = static_cast<Uint8>(255.99f * glm::clamp(color.r, 0.0f, 1.0f));
    Uint8 g = static_cast<Uint8>(255.99f * glm::clamp(color.g, 0.0f, 1.0f));
    Uint8 b = static_cast<Uint8>(255.99f * glm::clamp(color.b, 0.0f, 1.0f));
    return (255 << 24) | (b << 16) | (g << 8) | r;
}

struct RenderTask {
    int thread_id;
    int y_start;
    int y_end;
};

void render_thread_func(RenderTask task) 
{
    while (g_running.load()) 
    {
        if (g_frame_ready)
            continue;

        int buffer = g_current_buffer.load();
        Uint32* pixels = g_pixels[buffer].data();

        int width = g_render_width.load();
        int height = g_render_height.load();

        Camera::CameraBasis basis = g_camera_basis;

        for (int y = task.y_start; y < task.y_end; y++) 
        {
            for (int x = 0; x < width; x++) 
            {
                float screen_x = (2.0f * x - width) / float(height);
                float screen_y = (float(height) - 2.0f * y) / float(height);

                Ray ray = g_camera->GetRayForPixel(basis, screen_x, screen_y);

                Color color = TraceRay(ray, *g_world, 0);
                pixels[y * width + x] = ColorToPixel(color);
            }
        }

        int done = ++g_threads_done;

        if (done == NUM_THREADS) 
        {
            g_frame_ready.store(true);
        }

        std::unique_lock<std::mutex> lock(g_mutex);
        g_cv.wait_for(lock, std::chrono::milliseconds(0), [] {
            return !g_frame_ready.load() || !g_running.load();
            });
    }
}

bool Init() 
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0) 
    {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return false;
    }

    window = SDL_CreateWindow
    (
        "RayTracing - Solid Objects (F3 for GUI)",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        DEFAULT_WIDTH, 
        DEFAULT_HEIGHT,
        SDL_WINDOW_RESIZABLE
    );

    if (!window) 
    {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        return false;
    }

    renderer = SDL_CreateRenderer(window, -1, 0);
    if (!renderer)
    {
        SDL_Log("SDL_CreateRenderer failed: %s", SDL_GetError());
        return false;
    }

    texture = SDL_CreateTexture
    (
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        DEFAULT_WIDTH, 
        DEFAULT_HEIGHT
    );

    if (!texture)
    {
        SDL_Log("SDL_CreateTexture failed: %s", SDL_GetError());
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);

    g_pixels[0].resize(MAX_WIDTH * MAX_HEIGHT);
    g_pixels[1].resize(MAX_WIDTH * MAX_HEIGHT);

    Floor floor(-1.0f, Color(0.15f, 0.15f, 0.2f), Color(0.25f, 0.25f, 0.3f), 1.0f);
    g_world = new World(floor);

    g_world->AddLight(std::make_shared<Light>(Point3(0, 4.0f, -3.0f), 0.4f, Color(1.0f, 0.95f, 0.8f), 2.0f));
    g_world->AddLight(std::make_shared<Light>(Point3(-5.0f, 2.0f, -5.0f), 0.3f, Color(0.3f, 0.5f, 1.0f), 1.5f));
    g_world->AddLight(std::make_shared<Light>(Point3(5.0f, 1.5f, -6.0f), 0.35f, Color(1.0f, 0.6f, 0.3f), 1.5f));

    g_world->AddEntity(std::make_shared<Sphere>(Point3(0, 0.5f, -4.0f), 1.0f, Color(0.95f, 0.95f, 0.95f), 0.85f));

    float radius = 3.0f;
    int count = 6;
    for (int i = 0; i < count; i++) {
        float angle = (i * 2.0f * 3.14159f) / count;
        float x = std::cos(angle) * radius;
        float z = std::sin(angle) * radius - 4.0f;

        Color colors[] = {
            Color(1.0f, 0.3f, 0.3f), Color(0.3f, 1.0f, 0.3f),
            Color(0.3f, 0.3f, 1.0f), Color(1.0f, 1.0f, 0.3f),
            Color(1.0f, 0.3f, 1.0f), Color(0.3f, 1.0f, 1.0f),
        };

        float reflectivity[] = { 0.5f, 0.3f, 0.4f, 0.6f, 0.5f, 0.7f };
        float sizes[] = { 0.4f, 0.35f, 0.45f, 0.38f, 0.42f, 0.36f };

        g_world->AddEntity(std::make_shared<Sphere>(Point3(x, sizes[i], z), sizes[i], colors[i], reflectivity[i]));
    }

    auto ellip1 = std::make_shared<Ellipsoid>(Point3(-4.0f, 0.4f, -7.0f), Vec3(0.6f, 0.4f, 1.0f), Color(0.2f, 0.8f, 0.9f), 0.6f);
    ellip1->rotation_yaw = 30.0f;
    ellip1->UpdateRotationMatrix();
    g_world->AddEntity(ellip1);

    auto ellip2 = std::make_shared<Ellipsoid>(Point3(4.0f, 0.3f, -7.5f), Vec3(0.5f, 0.5f, 1.2f), Color(0.9f, 0.4f, 0.2f), 0.5f);
    ellip2->rotation_roll = 25.0f;
    ellip2->UpdateRotationMatrix();
    g_world->AddEntity(ellip2);

    auto box1 = std::make_shared<Box>(Point3(-3.0f, 0.5f, -5.0f), Vec3(0.5f, 0.5f, 0.5f), Color(0.4f, 0.4f, 0.9f), 0.3f);
    box1->rotation_yaw = 45.0f;
    box1->UpdateRotationMatrix();
    g_world->AddEntity(box1);

    auto box2 = std::make_shared<Box>(Point3(3.0f, 0.6f, -5.5f), Vec3(0.6f, 0.6f, 0.6f), Color(0.9f, 0.4f, 0.4f), 0.4f);
    box2->rotation_pitch = 15.0f;
    box2->rotation_yaw = -30.0f;
    box2->UpdateRotationMatrix();
    g_world->AddEntity(box2);

    float pyramid_z = -9.0f;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col <= row; col++) {
            float x = (col - row * 0.5f) * 1.2f;
            float y = row * 0.7f + 0.35f;
            float size = 0.35f - row * 0.04f;

            Color color;
            switch (row) {
            case 0: color = Color(1.0f, 0.9f, 0.4f); break;
            case 1: color = Color(0.4f, 1.0f, 0.9f); break;
            case 2: color = Color(1.0f, 0.4f, 1.0f); break;
            case 3: color = Color(1.0f, 0.6f, 0.4f); break;
            default: color = Color(1, 1, 1);
            }

            g_world->AddEntity(std::make_shared<Sphere>(Point3(x, y, pyramid_z), size, color, 0.5f));
        }
    }

    g_world->AddEntity(std::make_shared<Sphere>(Point3(-6.0f, 0.3f, -4.0f), 0.25f, Color(0.7f, 0.7f, 0.7f), 0.8f));
    g_world->AddEntity(std::make_shared<Sphere>(Point3(6.0f, 0.35f, -4.5f), 0.28f, Color(0.7f, 0.7f, 0.7f), 0.8f));

    g_camera = new Camera(Point3(0, 1.5f, 2.5f), 0.0f, -8.0f, 1.3f);

    g_camera_basis = g_camera->GetBasis();

    SDL_Log("Using %d threads for rendering", NUM_THREADS);
    SDL_Log("Scene: %d entities, %d lights", (int)g_world->entities.size(), (int)g_world->lights.size());

    return true;
}

std::vector<std::thread> g_threads;
std::vector<RenderTask> g_tasks;

Uint32 last_time = 0;
Uint32 last_frame_time = 0;

void tick()
{
    Uint32 current_time = SDL_GetTicks();

    float delta_time = (current_time - last_time) / 1000.0f;
    last_time = current_time;

    Uint32 frame_time = current_time - last_frame_time;
    Uint32 target_frame_time = 1000 / TARGET_FPS;

    if (frame_time < target_frame_time) 
    {
        Uint32 sleep_time = target_frame_time - frame_time;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));

        current_time = SDL_GetTicks();
        frame_time = current_time - last_frame_time;
    }
    last_frame_time = current_time;

    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL2_ProcessEvent(&event);

        if (event.type == SDL_QUIT) g_running.store(false);

        if (event.type == SDL_KEYDOWN) 
        {
            if (event.key.keysym.sym == SDLK_F3)
            {
                g_input.gui_visible = !g_input.gui_visible;
                if (g_input.gui_visible)
                {
                    g_input.mouse_captured = false;
                    SDL_SetRelativeMouseMode(SDL_FALSE);
                }
            }
            if (event.key.keysym.sym == SDLK_ESCAPE && g_input.gui_visible) g_running.store(false);
            if (event.key.keysym.sym == SDLK_w) g_input.w = true;
            if (event.key.keysym.sym == SDLK_a) g_input.a = true;
            if (event.key.keysym.sym == SDLK_s) g_input.s = true;
            if (event.key.keysym.sym == SDLK_d) g_input.d = true;
            if (event.key.keysym.sym == SDLK_LSHIFT || event.key.keysym.sym == SDLK_RSHIFT) g_input.shift = true;
        }

        if (event.type == SDL_KEYUP) 
        {
            if (event.key.keysym.sym == SDLK_w) g_input.w = false;
            if (event.key.keysym.sym == SDLK_a) g_input.a = false;
            if (event.key.keysym.sym == SDLK_s) g_input.s = false;
            if (event.key.keysym.sym == SDLK_d) g_input.d = false;
            if (event.key.keysym.sym == SDLK_LSHIFT || event.key.keysym.sym == SDLK_RSHIFT) g_input.shift = false;
        }

        if (event.type == SDL_MOUSEMOTION) HandleMouseEvent(event.motion);

        if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT)
        {
            if (!g_input.gui_visible) ToggleMouseCapture();
        }
    }

    ProcessInput(delta_time);

    ImGui_ImplSDLRenderer2_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    if (g_input.gui_visible) 
    {
        ImGui::Begin("Settings (F3 to hide)");

        ImGui::Text("Controls: WASD Move, Mouse Look, F3 Toggle GUI");

        ImGui::Separator();
        ImGui::Text("Lighting");
        ImGui::SliderFloat("Ambient Strength", &g_world->ambient_strength, 0.0f, 1.0f);

        ImGui::Separator();
        ImGui::Text("Ray Attenuation");
        ImGui::SliderFloat("Attenuation", &g_attenuation, 0.0f, 1.0f);

        ImGui::Separator();
        ImGui::Text("Entity Editor");

        g_entity_names.clear();
        for (const auto& entity : g_world->entities) 
        {
            g_entity_names.push_back(entity->name);
        }

        std::vector<const char*> entity_names_ptr;
        for (const auto& name : g_entity_names) 
        {
            entity_names_ptr.push_back(name.c_str());
        }

        int current_entity = g_selected_entity;
        if (ImGui::Combo("Select Entity", &current_entity, entity_names_ptr.data(), entity_names_ptr.size())) 
        {
            g_selected_entity = current_entity;
        }

        if (g_selected_entity >= 0 && g_selected_entity < (int)g_world->entities.size()) 
        {
            auto& entity = g_world->entities[g_selected_entity];

            ImGui::Text("Position:");
            ImGui::SliderFloat3("Center", &entity->center.x, -10.0f, 10.0f);

            ImGui::Separator();
            ImGui::PushID(0);
            ImGui::Text("Rotation:");
            ImGui::SliderFloat("Yaw", &entity->rotation_yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &entity->rotation_pitch, -90.0f, 90.0f);
            ImGui::SliderFloat("Roll", &entity->rotation_roll, -180.0f, 180.0f);
            ImGui::PopID();

            entity->UpdateRotationMatrix();

            ImGui::Separator();
            ImGui::Text("Color:");
            ImGui::ColorEdit3("Color", &entity->color.r);
            ImGui::SliderFloat("Reflectivity", &entity->reflectivity, 0.0f, 1.0f);

            ImGui::Separator();
            ImGui::Checkbox("Solid (Backface Culling)", &entity->solid);
        }

        ImGui::Separator();
        ImGui::Text("Render Resolution");
        ImGui::SliderInt("Width", &g_config_width, 32, 1920);
        ImGui::SliderInt("Height", &g_config_height, 24, 1080);

        if (ImGui::Button("Apply Resolution")) 
        {
            g_render_width.store(g_config_width);
            g_render_height.store(g_config_height);
            RecreateTexture();
        }

        ImGui::Separator();
        ImGui::Text("Presets");

        if (ImGui::Button("160x120")) 
        {
            g_config_width = 160;
            g_config_height = 120;
            g_render_width.store(g_config_width);
            g_render_height.store(g_config_height);
            RecreateTexture();
        }
        ImGui::SameLine();
        if (ImGui::Button("320x240")) 
        {
            g_config_width = 320;
            g_config_height = 240;
            g_render_width.store(g_config_width);
            g_render_height.store(g_config_height);
            RecreateTexture();
        }
        ImGui::SameLine();
        if (ImGui::Button("640x480")) 
        {
            g_config_width = 640;
            g_config_height = 480;
            g_render_width.store(g_config_width);
            g_render_height.store(g_config_height);
            RecreateTexture();
        }
        ImGui::SameLine();
        if (ImGui::Button("1024x768"))
        {
            g_config_width = 1024;
            g_config_height = 768;
            g_render_width.store(g_config_width);
            g_render_height.store(g_config_height);
            RecreateTexture();
        }
        ImGui::SameLine();
        if (ImGui::Button("1920x1080"))
        {
            g_config_width = 1920;
            g_config_height = 1080;
            g_render_width.store(g_config_width);
            g_render_height.store(g_config_height);
            RecreateTexture();
        }

        ImGui::Separator();
        ImGui::Text("Current: %dx%d", g_render_width.load(), g_render_height.load());

        ImGui::Separator();
        ImGui::Text("Camera");
        ImGui::SliderFloat("Position X", &g_camera->position.x, -10.0f, 10.0f);
        ImGui::SliderFloat("Position Y", &g_camera->position.y, -5.0f, 10.0f);
        ImGui::SliderFloat("Position Z", &g_camera->position.z, -10.0f, 10.0f);
        ImGui::SliderFloat("Yaw", &g_camera->yaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &g_camera->pitch, -90.0f, 90.0f);
        ImGui::SliderFloat("FOV", &g_camera->fov, 0.5f, 3.0f);

        ImGui::Separator();
        ImGui::Text("Performance");
        ImGui::Text("FPS: %.1f (Target: %d)", ImGui::GetIO().Framerate, TARGET_FPS);
        ImGui::Text("Threads: %d", NUM_THREADS);
        ImGui::Text("Pixels: %d", g_render_width.load() * g_render_height.load());
        ImGui::Text("Entities: %d", (int)g_world->entities.size());
        ImGui::Text("Lights: %d", (int)g_world->lights.size());

        ImGui::End();
    }
    else
    {
        ImGui::Begin("##Info", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
        ImGui::SetWindowPos(ImVec2(10, 10));
        ImGui::Text("FPS: %.1f | %dx%d | F3 for menu",
            ImGui::GetIO().Framerate,
            g_render_width.load(),
            g_render_height.load());
        ImGui::End();
    }


    g_camera_basis = g_camera->GetBasis();


    if (g_frame_ready.load()) {
        int buffer = g_current_buffer.load();
        int width = g_render_width.load();
        int height = g_render_height.load();

        SDL_UpdateTexture(texture, NULL, g_pixels[buffer].data(), width * sizeof(Uint32));

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        g_current_buffer.store(1 - buffer);
        g_frame_ready.store(false);
        g_threads_done.store(0);

        g_cv.notify_all();
    }

    ImGui::Render();
    ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);

    SDL_RenderPresent(renderer);
}

void start_thread_pool() {
    int height = g_render_height.load();
    int pixels_per_thread = height / NUM_THREADS;
    int remainder = height % NUM_THREADS;

    int y_current = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        RenderTask task;
        task.thread_id = i;
        task.y_start = y_current;
        task.y_end = (i == NUM_THREADS - 1) ? height : y_current + pixels_per_thread + (i < remainder ? 1 : 0);

        g_tasks.push_back(task);
        g_threads.emplace_back(render_thread_func, task);

        y_current = task.y_end;
    }
}

void Quit() {
    g_running.store(false);
    g_cv.notify_all();

    for (auto& thread : g_threads) {
        if (thread.joinable()) thread.join();
    }

    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    if (texture) SDL_DestroyTexture(texture);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (g_camera) delete g_camera;
    if (g_world) delete g_world;
    if (window) SDL_DestroyWindow(window);

    SDL_Quit();
}

int main(int argc, char* argv[]) {
    if (!Init()) return EXIT_FAILURE;

    start_thread_pool();

    last_time = SDL_GetTicks();
    last_frame_time = last_time;

    while (g_running.load()) 
    {
        tick();
    }

    Quit();
    return EXIT_SUCCESS;
}