# speech-lid

音声の先頭30秒から発話言語を99言語の中から判定します。OpenAI Whisper tiny 多言語版の言語判定部分を取り出し、10ビットの重みをGGUFとONNXで提供します。Rust版はWindows x64のCPUとVulkanで動作確認済みです。

元モデルはOpenAI公式の多言語版 `tiny.pt` です。元のコードとモデル重みはMITライセンス（Copyright (c) 2022 OpenAI）で公開されています。本プロジェクトでは追加学習を行わず、言語判定に必要な部分を抽出して変換しました。[公式リポジトリ](https://github.com/openai/whisper/tree/v20250625) · [元モデルのライセンス](model/LICENSE) · [モデルの作成手順とSHA-256](docs/model-conversion.md)

| モデル | ファイルサイズ | 使い方 |
|---|---:|---|
| [10ビットGGUF](model/lid10.gguf) | 23,403,136バイト | このリポジトリのRustライブラリ・CLIで使用。アーキテクチャは `whisper-lid10` |
| [10ビットONNX](model/lid.onnx) | 23,495,197バイト | ONNX Runtimeで使用。前処理用の係数と出力の言語順を `model/` に同梱 |

## Windowsで実行する

RustのMSVCツールチェーンとVisual C++ x64ランタイムを用意し、リポジトリのルートで実行してください。セットアップは、固定したggmlバイナリ（llama.cpp `b12413`）のSHA-256を照合し、必要なDLLを `target/release/` に配置します。Vulkanには対応GPUとドライバーが必要です。

```powershell
cargo build --release --locked
.\setup-runtime.ps1
.\target\release\speech-lid.exe --model model\lid10.gguf --backend cpu sample.wav
.\target\release\speech-lid.exe --model model\lid10.gguf --backend vulkan sample.mp3
```

結果は1ファイルにつきJSON 1行です。`language` は第1候補（例：日本語は `ja`）、`ranked` は上位2候補です。空の音声とデジタル無音では `status` が `no_speech` になります。複数ファイルは引数に並べるか、1行1パスの一覧を `--files-from` で渡すとモデルを使い回せます。

## Rustアプリに組み込む

アプリの隣にこのリポジトリを置く場合の依存設定です。16 kHz・モノラルのPCM（`f32`）を渡す場合は、音声デコーダーを省いてビルドできます。

```toml
[dependencies]
speech-lid = { path = "../lang_classification", default-features = false }
```

```rust
use speech_lid::{Backend, Detector};
use std::path::Path;

fn classify(model: &Path, plugins: &Path, clips: &[Vec<f32>])
    -> Result<(), Box<dyn std::error::Error>>
{
    let mut detector = Detector::load(model, Backend::Cpu, 8, plugins)?;
    for pcm in clips {
        if let Some(logits) = detector.predict(pcm)? {
            let (best, _) = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            println!("{}", detector.languages()[best]);
        }
    }
    Ok(())
}
```

`model` にGGUFのパスを指定します。アプリの実行ファイルと同じ場所に `ggml.dll`、`ggml-base.dll`、`libomp.dll` と使用する `cpu/` または `vulkan/` を配置し、その親ディレクトリを `plugins` に渡してください。GPUを使用する場合は `Backend::Vulkan` を選びます。

## ONNXの入出力

入力は `mel`（float32、`[1, 80, 3000]`）、出力は `logits`（float32、`[1, 99]`）です。最大値のインデックスを [言語表](model/languages.json) の `index` と照合してください。

前処理には [frontend.npz](model/frontend.npz) の `hann` と `mel` を使います。16 kHzのモノラルPCMを先頭480,000サンプルに切り詰め、短い場合は末尾をゼロ埋めします。両端を200サンプル反射パディングし、窓長400・ホップ160で実FFTのパワーを計算します。最後の1フレームを落とし、80バンドのメルフィルターを適用します。エネルギーの下限を `1e-10` として常用対数を取り、全体最大値から8を引いた値で下限をクリップし、最後に `(x + 4) / 4` で正規化します。

## ベンチマーク

Emilia-YODASの6言語各1,000件、計6,000件で、第1候補がデータ付属の言語ラベルと一致した件数です。評価対象の6言語に対し、判定時の候補は全99言語です。

| 言語 | ONNX CPU | GGUF CPU | GGUF Vulkan |
|---|---:|---:|---:|
| ドイツ語 | 993 | 993 | 993 |
| 英語 | 990 | 990 | 990 |
| フランス語 | 986 | 987 | 986 |
| 日本語 | 971 | 970 | 971 |
| 韓国語 | 990 | 990 | 990 |
| 中国語 | 955 | 955 | 956 |
| **合計** | **5,885（98.08%）** | **5,885（98.08%）** | **5,886（98.10%）** |

表は同じMP3を各実装の前処理で読んだ結果です。同一の16 kHz PCMを直接入力した場合、3条件とも5,885件正解でした。GGUF CPUは全6,000件でONNXと同じ言語を選びました。参考として、同じ評価セットで元のWhisper tinyは98.07%、4ビットONNXは97.48%、large-v3-turbo（GPU）は98.77%でした。

Core i7-13700KF / RTX 4070 Tiで、12件のウォームアップ後に同じ120ファイルを測定した結果です。処理時間にはデコード、リサンプル、特徴抽出、推論を含みます。

| 実装 | 中央値 | 95パーセンタイル | ピークRAM |
|---|---:|---:|---:|
| ONNX Runtime CPU・8スレッド（Python） | 94.6 ms | 135.1 ms | 746.9 MiB |
| Rust GGUF CPU・8スレッド | 72.8 ms | 88.4 ms | 121.7 MiB |
| Rust GGUF Vulkan | 11.1 ms | 32.7 ms | 226.7 MiB（VRAM別） |

## ライセンス

このプロジェクトのコードは [MIT](LICENSE)、Whisper由来のモデルと前処理データには [OpenAIのMITライセンス](model/LICENSE) が適用されます。ggml・LLVM OpenMPの通知、音声デコーダーSymphoniaのMPL-2.0、Rust依存関係のライセンス一覧は [licenses/](licenses/) にあります。
