通义千问-图像编辑模型（qwen-image-edit-plus）支持多图输入和多图输出，可精确修改图内文字、增删或移动物体、改变主体动作、迁移图片风格及增强画面细节。

**快速入口：**[使用指南](https://help.aliyun.com/zh/model-studio/qwen-image-edit-guide) **|** [技术博客](https://qwen.ai/blog?id=1675c295dc29dd31073e5b3f72876e9d684e41c6&from=research.research-list) | [在线体验](https://bailian.console.aliyun.com/?tab=model#/efm/model_experience_center/vision?currentTab=imageGenerate&modelId=qwen-image-edit)


**模型名称**

**模型简介**

**输出图像规格**

qwen-image-edit-plus`**推荐**`

> 当前与qwen-image-edit-plus-2025-10-30能力相同

qwen-image-edit-plus系列模型，支持单图编辑和多图融合。

*   可输出 **1-6** 张图片。
    
*   支持自定义分辨率。
    
*   支持提示词智能优化**。**
    

**格式**：PNG  
**分辨率**：

*   **可指定**：通过 `[parameters.size](https://help.aliyun.com/zh/model-studio/qwen-image-edit-api?mode=pure#0e360df4915xx)` 参数指定输出图像的`宽*高`（单位：像素）。
    
*   **默认（不指定时）**：总像素接近 `1024*1024`，宽高比与输入图（多图输入时为最后一张）一致。
    

qwen-image-edit-plus-2025-12-15 `**推荐**`

qwen-image-edit-plus-2025-10-30 `**推荐**`

qwen-image-edit

支持单图编辑和多图融合。

*   仅支持输出 1 张图片。
    
*   不支持自定义分辨率。
    

**格式**：PNG

**分辨率**：**不可指定**。生成规则同上方的**默认**规则。

**说明**

调用前，请查阅各地域支持的[模型列表与价格](https://help.aliyun.com/zh/model-studio/models#bfe15d8aa2lxh)。

**计费说明：**

*   按成功生成的 **图像张数** 计费（单次请求如果返回n张图片，则当次费用为 n×单价）。模型调用失败或处理错误不产生任何费用，也不消耗[免费额度](https://help.aliyun.com/zh/model-studio/new-free-quota)。
    
*   您可开启“免费额度用完即停”功能，以避免免费额度耗尽后产生额外费用。详情请参见[免费额度](https://help.aliyun.com/zh/model-studio/new-free-quota)。
    

HTTP调用
------

在调用前，您需要[获取与配置 API Key](https://help.aliyun.com/zh/model-studio/get-api-key)，再[配置API Key到环境变量](https://help.aliyun.com/zh/model-studio/configure-api-key-through-environment-variables)。

如需通过SDK进行调用，请[安装DashScope SDK](https://help.aliyun.com/zh/model-studio/install-sdk)。目前，该SDK已支持Python和Java。

**重要**

北京和新加坡地域拥有独立的 **API Key** 与**请求地址**，不可混用，跨地域调用将导致鉴权失败或服务报错。

**北京地域**：`POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation`

**新加坡地域**：`POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation`

#### 请求参数

单图编辑

多图融合

此处以使用`qwen-image-edit-plus`模型输出2张图片为例。

以下为北京地域 URL ，若使用新加坡地域的模型，需将 URL 替换为：`https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation`

    curl --location 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation' \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer $DASHSCOPE_API_KEY" \
    --data '{
        "model": "qwen-image-edit-plus",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/fpakfo/image36.webp"
                        },
                        {
                            "text": "生成一张符合深度图的图像，遵循以下描述：一辆红色的破旧的自行车停在一条泥泞的小路上，背景是茂密的原始森林"
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "n": 2,
            "negative_prompt": "低质量",
            "prompt_extend": true,
            "watermark": false
        }
    }'

##### 请求头（Headers）

**Content-Type** `_string_` **（必选）**

请求内容类型。此参数必须设置为`application/json`。

**Authorization** `_string_`**（必选）**

请求身份认证。接口使用阿里云百炼API-Key进行身份认证。示例值：Bearer sk-xxxx。

##### 请求体（Request Body）

**model** `_string_` **（必选）**

模型名称，可选以下模型：

qwen-image-edit-plus系列模型：包括 `qwen-image-edit-plus`、`qwen-image-edit-plus-2025-12-15`、`qwen-image-edit-plus-2025-10-30`，支持输出1-6张图片。

`qwen-image-edit`：仅支持输出1张图片。

**input** `_object_` **（必选）**

输入参数对象，包含以下字段：

**属性**

**messages** `_array_` **（必选）**

请求内容数组。**当前仅支持单轮对话**，因此数组内**有且只有一个对象**，该对象包含`role`和`content`两个属性。

**属性**

**role** `_string_` **（必选）**

消息发送者角色，必须设置为`user`。

**content** `_array_` **（必选）**

消息内容，包含1-3张图像，格式为 `{"image": "..."}`；以及单个编辑指令，格式为 `{"text": "..."}`。

**属性**

**image** `_string_` **（必选）**

输入图像的 URL 或 Base64 编码数据。支持传入1-3张图像。

多图输入时，按照数组顺序定义图像顺序，输出图像的比例以最后一张为准。

**图像要求：**

*   图像格式：JPG、JPEG、PNG、BMP、TIFF、WEBP和GIF。
    
    > 输出图像为PNG格式，对于GIF动图，仅处理其第一帧。
    
*   图像分辨率：为获得最佳效果，建议图像的宽和高均在384像素至3072像素之间。分辨率过低可能导致生成效果模糊，过高则会增加处理时长。
    
*   图像大小：不超过10MB。
    

**支持的输入格式：**

1.  公网可访问的URL
    
    *   支持 HTTP 或 HTTPS 协议。本地文件请参见[上传文件获取临时 URL](https://help.aliyun.com/zh/model-studio/get-temporary-file-url)。
        
    *   示例值：`https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/fpakfo/image36.webp`
        
2.  传入 Base64 编码图像后的字符串
    
    *   示例值：`data:image/jpeg;base64,GDU7MtCZz...`（示例已截断，仅做演示）
        
    *   Base64 编码规范请参见[通过Base64编码传入图片](https://help.aliyun.com/zh/model-studio/qwen-image-edit-api?mode=pure#907c84c1a6wrm)。
        

**text** `_string_` **（必选）**

图像编辑指令，即正向提示词，用来描述生成图像中期望包含的元素和视觉特点。

进行多图像编辑时，编辑指令中需要使用“图1”、“图2”、“图3”等描述来指代相应的图片，否则会出现不符合预期的编辑结果。

支持中英文，长度上限800个字符，每个汉字/字母占一个字符，超过部分会自动截断。

示例值：图1中的女生穿着图2中的黑色裙子按图3的姿势坐下，保持其服装、发型和表情不变，动作自然流畅。

**parameters** `_object_` （可选）

控制图像生成的附加参数。

**属性**

**n** `_integer_` （可选）

输出图像的数量，默认值为1。

对于qwen-image-edit-plus系列模型，可选择输出1-6张图片。

对于`qwen-image-edit`，仅支持输出1张图片。

**negative\_prompt** `_string_` （可选）

反向提示词，用来描述不希望在画面中看到的内容，可以对画面进行限制。

支持中英文，长度上限500个字符，每个汉字/字母占一个字符，超过部分会自动截断。

示例值：低分辨率、错误、最差质量、低质量、残缺、多余的手指、比例不良等。

**size** `_string_` （可选）

设置输出图像的分辨率，格式为`宽*高`，例如`"1024*2048"`。宽和高的取值范围均为\[512, 2048\]像素。

**默认行为**：若不设置，输出图像将保持与输入图像（多图输入时为最后一张）相似的长宽比_，_接近`1024*1024`分辨率。

**支持模型**：仅qwen-image-edit-plus系列模型支持。

**prompt\_extend** `_bool_` （可选）

是否开启prompt智能改写。开启后，将使用大模型优化正向提示词，对描述性不足、较为简单的prompt提升效果较明显。

*   `true`：**默认值**，开启智能改写。
    
*   `false`：不开启智能改写。
    

**支持模型**：仅qwen-image-edit-plus系列模型支持。

**watermark** `_bool_` （可选）

是否在图像右下角添加 "Qwen-Image" 水印。默认值为 `false`。水印样式如下：

![1](https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/8972029571/p1012089.jpg)

**seed** `_integer_` （可选）

随机数种子，取值范围`[0,2147483647]`。

使用相同的`seed`参数值可使生成内容保持相对稳定。若不提供，算法将自动使用随机数种子。

**注意**：模型生成过程具有概率性，即使使用相同的`seed`，也不能保证每次生成结果完全一致。

#### 响应参数

任务执行成功

任务执行异常

任务数据（如任务状态、图像URL等）仅保留24小时，超时后会被自动清除。请您务必及时保存生成的图像。

    {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                            },
                            {
                                "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                            }
                        ]
                    }
                }
            ]
        },
        "usage": {
            "width": 1248,
            "image_count": 2,
            "height": 832
        },
        "request_id": "bf37ca26-0abe-98e4-8065-xxxxxx"
    }

**output** `_object_`

包含模型生成结果。

**属性**

**choices** `_array_`

结果选项列表。

**属性**

**finish\_reason** `_string_`

任务停止原因，自然停止时为`stop`。

**message** `_object_`

模型返回的消息。

**属性**

**role** `_string_`

消息的角色，固定为`assistant`。

**content** `_array_`

消息内容，包含生成的图像信息。

**属性**

**image** `_string_`

生成图像的 URL，格式为PNG。**链接有效期为24小时**，请及时下载并保存图像。

**usage** `_object_`

本次调用的资源使用情况，仅调用成功时返回。

**属性**

**image\_count** `_integer_`

生成的图像数量，等于选择输出图片的数量。

**width** `_integer_`

生成图像的宽度（像素）。

**height** `_integer_`

生成图像的高度（像素）。

**request\_id** `_string_`

请求唯一标识。可用于请求明细溯源和问题排查。

**code** `_string_`

请求失败的错误码。请求成功时不会返回此参数，详情请参见[错误信息](https://help.aliyun.com/zh/model-studio/error-code)。

**message** `_string_`

请求失败的详细信息。请求成功时不会返回此参数，详情请参见[错误信息](https://help.aliyun.com/zh/model-studio/error-code)。

DashScope SDK调用
---------------

SDK 的参数命名与[HTTP接口](https://help.aliyun.com/zh/model-studio/qwen-image-edit-api?mode=pure#42703589880ts)基本一致，参数结构根据语言特性进行封装，完整参数列表请参见[通义千问 API 参考](https://help.aliyun.com/zh/model-studio/qwen-api-reference)。

### Python SDK调用

**说明**

*   推荐安装最新版DashScope Python SDK，否则可能运行报错：[安装或升级SDK](https://help.aliyun.com/zh/model-studio/install-sdk)。
    
*   不支持异步接口。
    

#### **请求示例**

此处以使用`qwen-image-edit-plus`模型输出2张图片为例。

通过公网URL传入图片

通过Base64编码传入图片

通过URL下载图像

    import json
    import os
    from dashscope import MultiModalConversation
    import dashscope
    
    # 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    
    # 模型支持输入1-3张图片
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/thtclx/input1.png"},
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/iclsnx/input2.png"},
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/gborgw/input3.png"},
                {"text": "图1中的女生穿着图2中的黑色裙子按图3的姿势坐下"}
            ]
        }
    ]
    
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼 API Key 将下行替换为：api_key="sk-xxx"
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    # qwen-image-edit-plus支持输出1-6张图片，此处以2张为例
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen-image-edit-plus",
        messages=messages,
        stream=False,
        n=2,
        watermark=False,
        negative_prompt="低质量",
        prompt_extend=True,
        # 仅当输出图像数量n=1时支持设置size参数，否则会报错
        # size="1024*2048",
    )
    
    if response.status_code == 200:
        # 如需查看完整响应，请取消下行注释
        # print(json.dumps(response, ensure_ascii=False))
        for i, content in enumerate(response.output.choices[0].message.content):
            print(f"输出图像{i+1}的URL:{content['image']}")
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")
    

#### **响应示例**

图像链接的有效期为24小时，请及时下载图像。

> `input_tokens`和`output_tokens`为兼容字段，当前固定为0。

    {
        "status_code": 200,
        "request_id": "121d8c7c-360b-4d22-a976-6dbb8bxxxxxx",
        "code": "",
        "message": "",
        "output": {
            "text": null,
            "finish_reason": null,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                            },
                            {
                                "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                            }
                        ]
                    }
                }
            ]
        },
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "height": 1248,
            "image_count": 2,
            "width": 832
        }
    }

### Java SDK调用

**说明**

推荐安装最新版DashScope Java SDK，否则可能运行报错：[安装或升级SDK](https://help.aliyun.com/zh/model-studio/install-sdk)。

#### **请求示例**

此处以使用`qwen-image-edit-plus`模型输出2张图片为例。

通过公网URL传入图片

通过Base64编码传入图片

通过URL下载图像

    package org.example;
    
    import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversation;
    import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationParam;
    import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationResult;
    import com.alibaba.dashscope.common.MultiModalMessage;
    import com.alibaba.dashscope.common.Role;
    import com.alibaba.dashscope.exception.ApiException;
    import com.alibaba.dashscope.exception.NoApiKeyException;
    import com.alibaba.dashscope.exception.UploadFileException;
    import com.alibaba.dashscope.utils.Constants;
    import com.alibaba.dashscope.utils.JsonUtils;
    
    import java.io.IOException;
    import java.util.*;
    
    public class QwenImageEdit {
    
        static {
            // 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
            Constants.baseHttpApiUrl = "https://dashscope.aliyuncs.com/api/v1";
        }
    
        // 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        // 若没有配置环境变量，请用百炼 API Key 将下行替换为：apiKey="sk-xxx"
        static String apiKey = System.getenv("DASHSCOPE_API_KEY");
    
        public static void call() throws ApiException, NoApiKeyException, UploadFileException, IOException {
    
            MultiModalConversation conv = new MultiModalConversation();
    
            // 模型支持输入1-3张图片
            MultiModalMessage userMessage = MultiModalMessage.builder().role(Role.USER.getValue())
                    .content(Arrays.asList(
                            Collections.singletonMap("image", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/thtclx/input1.png"),
                            Collections.singletonMap("image", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/iclsnx/input2.png"),
                            Collections.singletonMap("image", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/gborgw/input3.png"),
                            Collections.singletonMap("text", "图1中的女生穿着图2中的黑色裙子按图3的姿势坐下")
                    )).build();
            // qwen-image-edit-plus支持输出1-6张图片，此处以2张为例
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("watermark", false);
            parameters.put("negative_prompt", "低质量");
            parameters.put("n", 2);
            parameters.put("prompt_extend", true);
            // 仅当输出图像数量n=1时支持设置size参数，否则会报错
            // parameters.put("size", "1024*2048");
    
            MultiModalConversationParam param = MultiModalConversationParam.builder()
                    .apiKey(apiKey)
                    .model("qwen-image-edit-plus")
                    .messages(Collections.singletonList(userMessage))
                    .parameters(parameters)
                    .build();
    
            MultiModalConversationResult result = conv.call(param);
            // 如需查看完整响应，请取消下行注释
            // System.out.println(JsonUtils.toJson(result));
            List<Map<String, Object>> contentList = result.getOutput().getChoices().get(0).getMessage().getContent();
            int imageIndex = 1;
            for (Map<String, Object> content : contentList) {
                if (content.containsKey("image")) {
                    System.out.println("输出图像" + imageIndex + "的URL：" + content.get("image"));
                    imageIndex++;
                }
            }
        }
    
        public static void main(String[] args) {
            try {
                call();
            } catch (ApiException | NoApiKeyException | UploadFileException | IOException e) {
                System.out.println(e.getMessage());
            }
        }
    }

#### **响应示例**

图像链接的有效期为24小时，请及时下载图像。

    {
        "requestId": "46281da9-9e02-941c-ac78-be88b8xxxxxx",
        "usage": {
            "image_count": 2,
            "width": 1216,
            "height": 864
        },
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                            },
                            {
                                "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                            }
                        ]
                    }
                }
            ]
        }
    }

**图像访问权限配置**
------------

模型生成的图像存储于阿里云OSS，每张图像会被分配一个OSS链接，如`https://dashscope-result-xx.oss-cn-xxxx.aliyuncs.com/xxx.png`。OSS链接允许公开访问，可以使用此链接查看或者下载图片，链接仅在 24 小时内有效。

如果您的业务对安全性要求较高，无法访问阿里云OSS链接，则需要单独配置外网访问白名单。请将以下域名添加到您的白名单中，以便顺利访问图片链接。

    dashscope-result-bj.oss-cn-beijing.aliyuncs.com
    dashscope-result-hz.oss-cn-hangzhou.aliyuncs.com
    dashscope-result-sh.oss-cn-shanghai.aliyuncs.com
    dashscope-result-wlcb.oss-cn-wulanchabu.aliyuncs.com
    dashscope-result-zjk.oss-cn-zhangjiakou.aliyuncs.com
    dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com
    dashscope-result-hy.oss-cn-heyuan.aliyuncs.com
    dashscope-result-cd.oss-cn-chengdu.aliyuncs.com
    dashscope-result-gz.oss-cn-guangzhou.aliyuncs.com
    dashscope-result-wlcb-acdr-1.oss-cn-wulanchabu-acdr-1.aliyuncs.com

**错误码**
-------

如果模型调用失败并返回报错信息，请参见[错误信息](https://help.aliyun.com/zh/model-studio/error-code)进行解决。

**常见问题**
--------

#### **Q：qwen-image-edit 支持多轮对话式编辑吗？**

A：不支持。模型仅支持单轮执行。每次调用均为独立、无状态的任务。如需连续编辑，须将生成的图片作为新输入再次调用。

#### **Q：qwen-image-edit 和 qwen-image-edit-plus 系列模型支持哪些语言？**

A：目前正式支持**简体中文和英文**；其他语言可自行尝试，但效果未经充分验证，可能存在不确定性。

#### **Q：**上传多张不同比例的参考图时，输出图像的比例以哪张为准？

A：输出图像会以**最后一张**上传的参考图的比例为准。

#### **Q：如何查看模型调用量？**

A：模型的调用信息存在小时级延迟，在模型调用完一小时后，请在模型观测（[北京](https://bailian.console.aliyun.com/#/model-telemetry)或[新加坡](https://modelstudio.console.aliyun.com/#/model-telemetry)）页面，查看调用量、调用次数、成功率等指标。详情请参见[如何查看模型调用记录](https://help.aliyun.com/zh/model-studio/new-free-quota#ab6ba5c538rn3)。