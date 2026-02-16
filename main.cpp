#include <algorithm>
#include <SFML/Graphics.hpp>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include <omp.h>

// Screen and Fractal constant
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 960;
constexpr int MAX_ITER = 255;
constexpr float R_MIN = -2.0f, R_MAX = 0.5f;
constexpr float I_MIN = -1.2f, I_MAX = 1.2f;
constexpr int N_PALETTE = 1;
constexpr int S = 2;
constexpr bool AVX512MODE = false;

inline int calculate_pixel(float cr, float ci) {
    float zr = 0.0f, zi = 0.0f;
    for (int count = 0; count < MAX_ITER; ++count) {
        float zr2 = zr * zr;
        float zi2 = zi * zi;
        if (zr2 + zi2 > 4.0f) return count;

        zi = 2.0f * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    return MAX_ITER;
}

sf::Color get_color(int n) {
    if (n == MAX_ITER) return sf::Color::Black;

    static const sf::Color palette[] = {
        {66, 30, 15},   // Dark Brown
        {25, 7, 26},    // Dark Purple
        {9, 1, 47},     // Deep Blue
        {4, 4, 73},     // Navy
        {0, 7, 100},    // Blue
        {12, 44, 138},  // Light Blue
        {24, 82, 177},  // Sky Blue
        {57, 125, 209}, // Pale Blue
        {134, 181, 229},// White-Blue
        {211, 236, 248},// Almost White
        {241, 233, 191},// Warm White
        {248, 201, 95}, // Sand
        {255, 170, 0},  // Gold
        {204, 128, 0},  // Orange
        {153, 87, 0},   // Rust
        {106, 52, 3}    // Brown
    };

    return palette[n % 16];
}
void render_avx512(uint8_t* pixels) {
    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < HEIGHT; ++y) {
        float ci_val = I_MIN + (float(y) / HEIGHT) * (I_MAX - I_MIN);
        __m512 ci = _mm512_set1_ps(ci_val);

        for (int x = 0; x < WIDTH; x += 16) {
            float start_r = R_MIN + (float(x) / WIDTH) * (R_MAX - R_MIN);
            float step_r = (1.f / WIDTH) * (R_MAX - R_MIN);


            __m512 cr = _mm512_add_ps(_mm512_set1_ps(start_r),
                _mm512_mul_ps(_mm512_setr_ps(0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.)
                    ,_mm512_set1_ps(step_r))
            );

            __m512 zr = _mm512_setzero_ps();
            __m512 zi = _mm512_setzero_ps();

            // These will store the magnitude used for final coloring
            __m512 final_zr2 = _mm512_setzero_ps();
            __m512 final_zi2 = _mm512_setzero_ps();

            __m512i iterations = _mm512_setzero_si512();
            __m512 four = _mm512_set1_ps(4.0f);

            // Start with all lanes active
            __mmask16 active_mask = 0xFFFF;

            for (int i = 0; i < MAX_ITER; ++i) {
                __m512 current_zr2 = _mm512_mul_ps(zr, zr);
                __m512 current_zi2 = _mm512_mul_ps(zi, zi);
                __m512 mag = _mm512_add_ps(current_zr2, current_zi2);

                final_zr2 = _mm512_mask_mov_ps(final_zr2, active_mask, current_zr2);
                final_zi2 = _mm512_mask_mov_ps(final_zi2, active_mask, current_zi2);

                __mmask16 inside_mask = _mm512_cmp_ps_mask(mag, four, _CMP_LT_OQ);

                active_mask = _kand_mask16(active_mask, inside_mask);

                // If all lanes escaped, stop.
                if (active_mask == 0) break;

                iterations = _mm512_mask_add_epi32(iterations, inside_mask, iterations, _mm512_set1_epi32(1));

                __m512 next_zi = _mm512_fmadd_ps(_mm512_add_ps(zr, zr), zi, ci);
                zr = _mm512_add_ps(_mm512_sub_ps(current_zr2, current_zi2), cr);
                zi = next_zi;
            }

            int32_t counts[16];
            float store_zr2[16], store_zi2[16];
            _mm512_storeu_si512(counts, iterations);
            _mm512_storeu_ps(store_zi2, final_zi2); // Use the fixed final variables
            _mm512_storeu_ps(store_zr2, final_zr2);

            for (int i = 0; i < 16; ++i) {
                int idx = (y * WIDTH + (x + i)) * 4;
                int n = counts[i];
                if (n >= MAX_ITER) {
                    pixels[idx] = 0; pixels[idx+1]=0; pixels[idx+2]=0;
                } else {
                    float mag_sq = store_zr2[i] + store_zi2[i];

                    if (mag_sq <= 1.0f) mag_sq = 1.0001f;

                    float smooth_n = static_cast<float>(n) + 1.0f - std::log2f(0.5f * std::log2f(mag_sq));

                    sf::Color c1 = get_color(static_cast<int>(smooth_n));
                    sf::Color c2 = get_color(static_cast<int>(smooth_n) + 1);
                    float frac = smooth_n - std::floor(smooth_n);

                    pixels[idx] = static_cast<uint8_t>(c1.r + frac * (c2.r - c1.r));
                    pixels[idx+1] = static_cast<uint8_t>(c1.g + frac * (c2.g - c1.g));
                    pixels[idx+2] = static_cast<uint8_t>(c1.b + frac * (c2.b - c1.b));
                }
                pixels[idx+3] = 255;
            }
        }
    }
}
int main() {
    //One thread per core is better than 2.
    omp_set_num_threads(std::thread::hardware_concurrency() / 2);
    std::vector<uint8_t> pixels(WIDTH * HEIGHT * 4);

    //To measure long-term performance instead of short-term performance.
    std::cout << "Warming up CPU..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        render_avx512(pixels.data());
    }

    long long total_checksum = 0;

    constexpr int NUM_TRIALS = 500;
    std::vector<double> times;
    times.reserve(NUM_TRIALS);

    std::cout << "Benchmarking " << NUM_TRIALS << " frames..." << std::endl;

    for (int i = 0; i < NUM_TRIALS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        render_avx512(pixels.data());

        //To prevent compiler optimization
        total_checksum += pixels[0] + pixels[WIDTH * HEIGHT / 2] + pixels[WIDTH * HEIGHT / 2];
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        times.push_back(elapsed_seconds.count());
    }
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / NUM_TRIALS;

    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / NUM_TRIALS - mean * mean);
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Average Time:  " << mean << "s (" << (1.0/mean) << " FPS)" << std::endl;
    std::cout << "Min Time:      " << min_time << "s" << std::endl;
    std::cout << "Max Time:      " << max_time << "s" << std::endl;
    std::cout << "Variance:      " << (stdev / mean * 100.0) << "% deviation" << std::endl;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Mandelbrot Optimization Project");
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    texture.update(pixels.data());
    sf::Sprite sprite(texture);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }
        window.clear();
        window.draw(sprite);
        window.display();
    }
    return 0;
}
//benchmarked with no power supply. Test spec
//Lenovo Ideapad slim 5(16inch, amd gen 10)
//RYZEN AI 7 350(laptop)
//DDR5 32GB 5600MT
//PCIe4.0 512GB TLC SSD

//226FPS(AVX-512)
//#pragma omp parallel for schedule(guided) thread imbalance is high. (gomp_team_barrier_wait_end 17.2% on prof)
//changed to schedule(dynamic, n)
//n = 4: 293FPS | 290FPS (gomp_team_barrier_wait_end dropped below 1%)
//n = 1: 297FPS | 291FPS | 298FPS
//n = 16: 293FPS | 280FPS | 294FPS
//n = 64: 240FPS | 236FPS | 229FPS | 232FPS
//conclusion: n = 1 ~ 16 irrelevent. I'll use n = 4;

//no what's the problem?
//AVX512_render consume 60%. Gemini says I can raise it to over 85%.
//changing coloring to AVX512 later.