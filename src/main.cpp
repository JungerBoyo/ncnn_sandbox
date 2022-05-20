#include <net.h> // ncnn
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#include <spdlog/spdlog.h>
#include <string_view>
#include <fstream>
#include <filesystem>
#include <project_config/config.hpp>

namespace fs = std::filesystem;

void normalize(ncnn::Mat& img);
void produceDBGImg(const ncnn::Mat& img);

int main()
{
  int w;
  int h;
  int chNum;

  constexpr std::string_view ImgPath = "static_res/imgs/test2.png";
  constexpr std::string_view NcnnModelsPath = "static_res/ncnnModels/";
  constexpr int reqW{ 28 };
  constexpr int reqH{ 28 };

  fs::path projectDirPath(cmake::PROJECT_DIR);

  auto imgPath = fs::path(projectDirPath / fs::path(ImgPath));
  auto* imgPtr = stbi_load(imgPath.c_str(), &w, &h, &chNum, 1);
  if(imgPtr == nullptr)
  {
    spdlog::error("failed to load img from {}", imgPath.c_str());
    return 1;
  }
  
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(imgPtr, ncnn::Mat::PIXEL_GRAY, w, h, reqW, reqH);
  stbi_image_free(imgPtr);

  ///// DEBUG 
  //in.to_pixels(imgPtr, ncnn::Mat::PIXEL_RGB);
  //stbi_write_png("static_res/imgs/test0_dbgout.png", reqW, reqH, 3, imgPtr, reqW * 3);
  /////
  
  //normalize(in);
  const auto mean { 0.f };
  const auto norm { 1/255.f };
  in.substract_mean_normalize(&mean, &norm);

  ncnn::Net net;
  net.opt.use_vulkan_compute = true;
  const auto fsNcnnModelsPath = fs::path(NcnnModelsPath);
  const auto paramsFilePath = fs::path(projectDirPath / fs::path(fsNcnnModelsPath / "fashion_MNIST_model.param"));
  const auto binFilePath = fs::path(projectDirPath / fs::path(fsNcnnModelsPath / "fashion_MNIST_model.bin"));

  if(net.load_param(paramsFilePath.c_str()) == -1)
  {
    spdlog::error("failed to load ncnn model params at {}", paramsFilePath.c_str());
    return 1;
  }

  if(net.load_model(binFilePath.c_str()) == -1)
  {
    spdlog::error("failed to load ncnn model bin at {}", paramsFilePath.c_str());
    return 1;
  }

  
  const auto apiVersion = net.vulkan_device()->info.api_version();
  spdlog::info("using vulkan device :: {}", net.vulkan_device()->info.device_name()); 
  spdlog::info("API version :: {}.{}.{}", 
    VK_API_VERSION_MAJOR(apiVersion),
    VK_API_VERSION_MINOR(apiVersion),
    VK_API_VERSION_PATCH(apiVersion)
  ); 

  ncnn::Extractor ex = net.create_extractor();
  ex.set_light_mode(true);
  //ex.set_num_threads(4);

  ex.input("in", in);

  ncnn::Mat out;
  ex.extract("out", out);

  for(int i{0}; i<out.total(); ++i)
  {
    spdlog::info("class_{} :: pred = {}", i, *out.row(i));
  }
}

void normalize(ncnn::Mat& img)
{
  auto mean{ 0.f };
  const auto norm{ 1/255.f };

  for(int q=0; q<img.c; ++q)
  {
    const float* ptr = img.channel(q); 
    for (int y=0; y<img.h; y++)
    {
      for (int x=0; x<img.w; x++)
      {
        mean += ptr[x];
      }
      ptr += img.w;
    }        
  } 

  mean /= static_cast<float>(img.total());

  img.substract_mean_normalize(&mean, &norm);
}

void produceDBGImg(const ncnn::Mat& img)
{
  std::string strImg = "";

  for(int q=0; q<img.c; ++q)
  {
    const float* ptr = img.channel(q); 
    for (int y=0; y<img.h; y++)
    {
      for (int x=0; x<img.w; x++)
      { 
        strImg += std::to_string(ptr[x]) + "  ";
      }
      strImg += "\n";
      ptr += img.w;
    }       

    strImg += "\n\n"; 
  } 

  std::ofstream ostr(fs::path(cmake::PROJECT_DIR) / fs::path("static_res/imgs/test1_dbg.txt"));
  ostr << strImg;
  ostr.close();
}
