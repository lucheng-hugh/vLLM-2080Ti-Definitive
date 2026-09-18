# 4xT10 Profile

## Qwen3.8-27B-FP8

测试权重：[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | 模式 | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---|---:|---|---:|---|---:|---:|---:|
| `qwen27b/w8a16/fast/dflash2-fp16kv-1x256k-text-only.env` | fast | 256K | FP16 | DFlash2（默认 K=7） | text-only | 299,474 | 1444.73 / 190.78 | 1456.77 / 188.94 |
| `qwen27b/w8a16/normal/nomtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | 无/自回归 | text+image | 356,764 | 1518.31 / 40.22 | 1563.00 / 37.70 |
| `qwen27b/w8a16/normal/mtp-fp16kv-1x256k-text-image.env` | normal | 256K | FP16 | MTP/3 | text+image | 315,343 | 1488.27 / 76.03 | 1514.79 / 77.53 |
| `qwen27b/w8a16/fast/mtp-tqk8v4-1x256k-text-image.env` | fast | 256K | TQK8V4 | MTP/3 | text+image | 797,912 | 1444.72 / 113.78 | 1497.69 / 85.95 |

## Qwen3.8-27B-NVFP4

测试权重：[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

| Profile | 模式 | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---|---:|---|---:|---|---:|---:|---:|
| `qwen27b/w4a16/fast/dflash2-fp16kv-2x256k-text-only.env` | fast | 2 x 256K | FP16 | DFlash2（默认 K=7） | text-only | 696,320 | 1703.67 / 442.12（C2 aggregate） | 1569.86 / 503.78（C2 aggregate） |

NVFP4 C2 行使用同步客户端提交。prefill 为两路 prompt token 总数除以最后一路 TTFT；decode 覆盖第一路首 token 到最后一路完成的完整窗口。同一服务的 C1 结果在 4K/128 下为 `1498.13 / 244.63`，在 32K/512 下为 `1354.99 / 253.20`。

数据来自四张 16 GiB Tesla T10（PCIe、TP=4）和 ABI 匹配的 PCIe CAR 扩展。`4K/128` 表示约 4K 输入、128 输出，`32K/512` 表示约 32K 输入、512 输出，两者都是高投机命中率下的合成测试。4K/128 取预热后三次请求的中位数，32K/512 为预热后一次请求；使用 `launcher.sh` 启动对应 profile 即可复测。

## TP/PP 混合布局验证

当前 launcher 也支持四张 T10 上的 `TP2 x PP2` 混合布局。2026-09-18 使用
GPU `0,2,3,4`、`--tensor-parallel-size 2` 和 `--pipeline-parallel-size 2` 启动
Qwen3.8-27B FP8 文本+图像路由；四个 worker 均成功就绪，`/health` 返回 HTTP 200。
单请求 smoke test 以及两个并发请求均以 HTTP 200 完成，`finish_reason=stop`，输出连贯，
未发现截断或重复。

这只是启动与正确性验证，不替代上面的 TP=4 性能数据。证据保存在验证机仓库外的
`/home/max/runtime-closure-20260918/tp2pp2/`。
