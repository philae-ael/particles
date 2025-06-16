#include <SDL3/SDL.h>
#include <SDL3/SDL_render.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <raylib.h>

struct AliveParticle {
  Vector2 position;
  Vector2 velocity;
  float lifetime;
};

struct Spawner {
  Vector2 position;
  Vector2 velocity;
  float spawnRate;
  float timeSinceLastSpawn;
};

struct DyingParticle {
  Vector2 position;
  float lifetime;
  float fadeOutTime = 1.0f;
};

struct Particles {
  SDL_Texture *texture;
  Spawner *spawners;
  std::size_t spawnerCount = 0;

  float *alivePosxBuffer = nullptr;
  float *alivePosyBuffer = nullptr;
  float *aliveVelxBuffer = nullptr;
  float *aliveVelyBuffer = nullptr;
  float *aliveLifetimeBuffer = nullptr;
  std::size_t aliveCount = 0;
  std::size_t aliveBufferSize = 0;

  float *deadLifetimeBuffer = nullptr;
  std::size_t deadCount = 0;
  std::size_t deadBufferSize = 0;

  float *tmpstorage = nullptr;
  std::size_t tmpstorageSize = 0;

  void run(float deltaTime) {
    auto newDeadCount = 0;

    {
      auto i = 0;
      auto d = _mm256_set1_ps(deltaTime);
      for (; i + 8 <= aliveCount; i += 8) {
        __m256 posx = _mm256_load_ps(alivePosxBuffer + i);
        __m256 velx = _mm256_load_ps(aliveVelxBuffer + i);

        posx = _mm256_fmadd_ps(velx, d, posx);
        _mm256_store_ps(alivePosxBuffer + i, posx);
      }
      i = 0;
      for (; i + 8 <= aliveCount; i += 8) {
        __m256 posy = _mm256_load_ps(alivePosyBuffer + i);
        __m256 vely = _mm256_load_ps(aliveVelyBuffer + i);
        posy = _mm256_fmadd_ps(vely, d, posy);
        _mm256_store_ps(alivePosyBuffer + i, posy);
      }
      i = 0;
      for (; i + 8 <= aliveCount; i += 8) {
        __m256 lifetime = _mm256_load_ps(aliveLifetimeBuffer + i);
        lifetime = _mm256_sub_ps(lifetime, d);
        _mm256_store_ps(aliveLifetimeBuffer + i, lifetime);

        auto mask = _mm256_cmp_ps(lifetime, _mm256_setzero_ps(), _CMP_LE_OS);
        auto deadMask = _mm256_movemask_ps(mask);

        // If the death rate is high,
        // my hypothesis is a second pass would be worth it.
        // First count, then grab the last N alive particles
        // and swap them with the dead particles in a pass
        // If the death rate is low,
        // this is probably better... we pay for a branch misprediction and
        // maybe a cache miss at each dead particle...
        newDeadCount += __builtin_popcount(deadMask);
      }

      for (; i < aliveCount; ++i) {
        alivePosxBuffer[i] += aliveVelxBuffer[i] * deltaTime;
        alivePosyBuffer[i] += aliveVelyBuffer[i] * deltaTime;

        if ((aliveLifetimeBuffer[i] -= deltaTime) <= 0) {
          newDeadCount++;
        }
      }
    }

    if (newDeadCount > 0) {
      if (tmpstorageSize < newDeadCount * 5) {
        tmpstorageSize = newDeadCount * 5;
        float *newStorage = new float[tmpstorageSize];
        delete[] tmpstorage;
        tmpstorage = newStorage;
      }
      // First move the last N alive particles into a temporary buffer
      auto i = aliveBufferSize - 1;
      for (auto j = 0; j < newDeadCount;) {
        if (i == 0) {
          break;
        }
        if (aliveLifetimeBuffer[i] > 0) {
          tmpstorage[j * 5 + 0] = alivePosxBuffer[i];
          tmpstorage[j * 5 + 1] = alivePosyBuffer[i];
          tmpstorage[j * 5 + 2] = aliveVelxBuffer[i];
          tmpstorage[j * 5 + 3] = aliveVelyBuffer[i];
          tmpstorage[j * 5 + 4] = aliveLifetimeBuffer[i];
          j++;
        }
        i--;
      }

      if (deadBufferSize < deadCount + newDeadCount) {
        deadBufferSize = (deadCount + newDeadCount) * 2;
        float *newBuffer =
            new (std::align_val_t(alignof(__m256))) float[deadBufferSize];
        delete[] deadLifetimeBuffer;
        deadLifetimeBuffer = newBuffer;
      }

      // Then remove the dead particles from the alive buffers and replace them
      // with the last N alive particles
      auto j = 0;
      for (std::size_t i = 0; i < aliveBufferSize; ++i) {
        if (j >= newDeadCount) {
          break;
        }
        deadLifetimeBuffer[deadCount + j] = aliveLifetimeBuffer[i];

        alivePosxBuffer[i] = tmpstorage[j * 5 + 0];
        alivePosyBuffer[i] = tmpstorage[j * 5 + 1];
        aliveVelxBuffer[i] = tmpstorage[j * 5 + 2];
        aliveVelyBuffer[i] = tmpstorage[j * 5 + 3];
        aliveLifetimeBuffer[i] = tmpstorage[j * 5 + 4];
        j++;
      }

      aliveCount -= newDeadCount;
      deadCount += newDeadCount;
    }

    for (auto i = 0; i < spawnerCount; ++i) {
      auto &spawner = spawners[i];
      spawner.timeSinceLastSpawn += deltaTime;
      if (spawner.timeSinceLastSpawn >= spawner.spawnRate) {
        float lifetime = SDL_randf() * 2.0f + 1.0f;
        Vector2 velocity = {SDL_randf() * 100.0f - 50.0f,
                            SDL_randf() * 100.0f - 50.0f};

        if (aliveBufferSize <= aliveCount) {
          aliveBufferSize = (aliveCount + aliveBufferSize) * 2;
          float *newBuffer =
              new (std::align_val_t(alignof(__m256))) float[aliveBufferSize];
          std::memcpy(newBuffer, alivePosxBuffer, aliveCount * sizeof(float));
          delete[] alivePosxBuffer;
          alivePosxBuffer = newBuffer;
          newBuffer =
              new (std::align_val_t(alignof(__m256))) float[aliveBufferSize];
          std::memcpy(newBuffer, alivePosyBuffer, aliveCount * sizeof(float));
          delete[] alivePosyBuffer;
          alivePosyBuffer = newBuffer;

          newBuffer =
              new (std::align_val_t(alignof(__m256))) float[aliveBufferSize];
          std::memcpy(newBuffer, aliveVelxBuffer, aliveCount * sizeof(float));
          delete[] aliveVelxBuffer;
          aliveVelxBuffer = newBuffer;
          newBuffer =
              new (std::align_val_t(alignof(__m256))) float[aliveBufferSize];
          std::memcpy(newBuffer, aliveVelyBuffer, aliveCount * sizeof(float));
          delete[] aliveVelyBuffer;

          aliveVelyBuffer = newBuffer;
          newBuffer =
              new (std::align_val_t(alignof(__m256))) float[aliveBufferSize];
          std::memcpy(newBuffer, aliveLifetimeBuffer,
                      aliveCount * sizeof(float));
          delete[] aliveLifetimeBuffer;
          aliveLifetimeBuffer = newBuffer;
        }

        alivePosxBuffer[aliveCount] = spawner.position.x;
        alivePosyBuffer[aliveCount] = spawner.position.y;
        aliveVelxBuffer[aliveCount] = velocity.x;
        aliveVelyBuffer[aliveCount] = velocity.y;
        aliveLifetimeBuffer[aliveCount] = lifetime;
        aliveCount++;

        spawner.timeSinceLastSpawn = 0;
      }
    }

    {
      for (std::size_t i = 0; i < deadCount; ++i) {
        auto &lifetime = deadLifetimeBuffer[i];
        lifetime -= deltaTime;

        if (lifetime <= 0) {
          deadLifetimeBuffer[i] = deadLifetimeBuffer[deadCount - 1];
          deadCount--;
          i--;
        }
      }
    }
  }

  // void draw(SDL_Renderer *renderer) {
  //   for (std::size_t i = 0; i < aliveCount; ++i) {
  //     auto &particle = aliveBuffer[i];
  //     SDL_FRect sourceRect = {0, 0, 32, 32};
  //     SDL_FRect destRect = {
  //         particle.position.x,
  //         particle.position.y,
  //         32.f / 8,
  //         32.f / 8,
  //     };
  //     SDL_RenderTexture(renderer, texture, &sourceRect, &destRect);
  //   }
  //
  //   for (auto i = 0; i < spawnerCount; ++i) {
  //     auto &spawner = spawners[i];
  //     SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
  //     SDL_RenderPoint(renderer, spawner.position.x, spawner.position.y);
  //   }
  //
  //   for (std::size_t i = 0; i < deadCount; ++i) {
  //     auto &particle = deadBuffer[i];
  //     float alpha = particle.lifetime / particle.fadeOutTime;
  //     SDL_SetRenderDrawColor(renderer, 255, 255, 255,
  //                            static_cast<Uint8>(alpha * 255));
  //     SDL_RenderPoint(renderer, particle.position.x, particle.position.y);
  //   }
  // }
};

int main(int argc, char *argv[]) {
  SDL_Init(SDL_INIT_VIDEO);

  auto window =
      SDL_CreateWindow("Particle System", 800, 600, SDL_WINDOW_RESIZABLE);
  if (!window) {
    SDL_Log("Could not create window: %s", SDL_GetError());
    return 1;
  }
  SDL_Renderer *renderer = SDL_CreateRenderer(window, nullptr);
  if (!renderer) {
    SDL_Log("Could not create renderer: %s", SDL_GetError());
    SDL_DestroyWindow(window);
    return 1;
  }

  SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                           SDL_TEXTUREACCESS_STATIC, 32, 32);
  unsigned char pixels[32 * 32 * 4];
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      int index = (y * 32 + x) * 4;
      if (x * x + y * y > 32 * 32) {
        pixels[index] = static_cast<unsigned char>(0);
        pixels[index + 1] = static_cast<unsigned char>(0);
        pixels[index + 2] = static_cast<unsigned char>(0);
        pixels[index + 3] = static_cast<unsigned char>(0);
      } else {
        pixels[index] = static_cast<unsigned char>(x * 8);
        pixels[index + 1] = static_cast<unsigned char>(y * 8);
        pixels[index + 2] = static_cast<unsigned char>(255);
        pixels[index + 3] = static_cast<unsigned char>(255);
      }
    }
  }
  SDL_UpdateTexture(texture, nullptr, pixels, 32 * 4);

  Particles particles{
      texture,
  };
  particles.spawnerCount = 500000;
  particles.spawners = new Spawner[particles.spawnerCount];
  for (int i = 0; i < particles.spawnerCount; ++i) {
    Vector2 position = {SDL_randf() * 800.0f, SDL_randf() * 600.0f};

    float spawnRate = SDL_randf() * 0.4f + 0.1f;

    particles.spawners[i] = {position, spawnRate, 0};
  }

  auto old_time = SDL_GetTicks();
  bool quit = false;
  while (!quit) {
    auto new_time = SDL_GetTicksNS();
    auto delta_time = (new_time - old_time) * 1e-9f;
    old_time = new_time;

    SDL_GL_SetSwapInterval(0);
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
      case SDL_EVENT_QUIT:
        quit = true;
        break;
      case SDL_EVENT_KEY_DOWN:
        if (event.key.key == SDLK_ESCAPE) {
          quit = true;
        }
        break;
      default:
        break;
      }
    }

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    particles.run(delta_time);
    // The slow part of the particle system is the draw call and i can't
    // instantiate the calls with SDL renderer...
    // particles.draw(renderer);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    char buffer[256];
    std::snprintf(
        buffer, 256, "Alive: %zu Dead: %zu, \ndt: %.02fms, log alive(10): %.1f",
        particles.aliveCount, particles.deadCount, delta_time * 1000.f,

        std::log(particles.aliveCount) / std::log(10));
    SDL_RenderDebugText(renderer, 10, 10, buffer);
    SDL_RenderPresent(renderer);
    SDL_UpdateWindowSurface(window);
  }

  SDL_DestroyTexture(texture);
  delete[] particles.spawners;
  delete[] particles.alivePosxBuffer;
  delete[] particles.alivePosyBuffer;
  delete[] particles.aliveVelxBuffer;
  delete[] particles.aliveVelyBuffer;
  delete[] particles.aliveLifetimeBuffer;
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  return 0;
}
