# speech-lid

音声の先頭30秒から発話言語を99言語の中から判定します。OpenAI Whisper tiny 多言語版の言語判定部分を取り出し、10ビット版とエンコーダーを8ビット化したCompact版をGGUFとONNXで提供します。Rust版はWindows x64のCPUとVulkanで動作確認済みです。

元モデルはOpenAI公式の多言語版 `tiny.pt` です。元のコードとモデル重みはMITライセンス（Copyright (c) 2022 OpenAI）で公開されています。本プロジェクトでは追加学習を行わず、言語判定に必要な部分を抽出して変換しました。[公式リポジトリ](https://github.com/openai/whisper/tree/v20250625) · [元モデルのライセンス](model/LICENSE) · [モデルの作成手順とSHA-256](docs/model-conversion.md)

| モデル | ファイルサイズ | 使い方 |
|---|---:|---|
| [10ビットGGUF](model/lid10.gguf) | 23,403,136バイト | このリポジトリのRustライブラリ・CLIで使用。アーキテクチャは `whisper-lid10` |
| [10ビットONNX](model/lid.onnx) | 23,495,197バイト | ONNX Runtimeで使用。前処理用の係数と出力の言語順を `model/` に同梱 |
| [校正付きランク96 GGUF](model/lid10-cal96.gguf) | 17,577,952バイト | Rust CPU・Vulkan用の小型版。アーキテクチャは `whisper-lid10-cal96` |
| [校正付きランク96 ONNX](model/lid10-cal96.onnx) | 17,646,296バイト | 小型版のONNX。入力・前処理・99言語の出力順は共通 |
| [Compact GGUF](model/lid-compact.gguf) | 15,018,304バイト | Rust CPU・Vulkan。エンコーダーのMLP幅は1,408、デコーダーはランク96 |
| [Compact ONNX](model/lid-compact.onnx) | 15,039,439バイト | ONNX Runtime CPU。自己注意に `com.microsoft.MultiHeadAttention` 演算子を使用 |

小型版ではデコーダーの中間変換を音声で校正した低ランク行列に置き換え、固定の計算をまとめました。パラメータ数は基準の言語分類版の15,953,664から11,675,904へ26.8%減りました。文字起こしを含む元の `tiny.pt`（37,184,640パラメータ）から数えると68.6%の削減です。[校正と変換の手順](docs/model-conversion.md#校正付きランク96版)

## Windowsで実行する

RustのMSVCツールチェーンとVisual C++ x64ランタイムを用意し、リポジトリのルートで実行してください。セットアップは、固定したggmlバイナリ（llama.cpp `b12413`）のSHA-256を照合し、必要なDLLを `target/release/` に配置します。Vulkanには対応GPUとドライバーが必要です。

```powershell
cargo build --release --locked
.\setup-runtime.ps1
.\target\release\speech-lid.exe --model model\lid10.gguf --backend cpu sample.wav
.\target\release\speech-lid.exe --model model\lid10.gguf --backend vulkan sample.mp3
.\target\release\speech-lid.exe --model model\lid10-cal96.gguf --backend cpu sample.wav
.\target\release\speech-lid.exe --model model\lid-compact.gguf --backend vulkan sample.mp3
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

### 基準の10ビット版

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

### 校正付きランク96版と基準版の比較

Emilia-YODASの同じ6,000件を16 kHz・モノラルのPCMにそろえ、録音ごとに前回の入力とSHA-256が一致することを確認した。判定候補は全99言語で、各言語の評価件数は1,000件。この6,000件にはランクの予備比較に使った600件を含む。低ランク行列の校正には別データのFLEURS開発分割を使用した。

| 正解件数 | ドイツ語 | 英語 | フランス語 | 日本語 | 韓国語 | 中国語 | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 基準の10ビット版（同一PCM） | 993 | 990 | 986 | 971 | 990 | 955 | 5,885（98.08%） |
| 校正付きランク96版（同一PCM） | 991 | 993 | 988 | 970 | 991 | 955 | 5,888（98.13%） |

小型版はGGUF CPU、GGUF Vulkan、ONNX CPUの3実装が全6,000件で同じ言語を選んだ。それぞれの基準版と比べ、誤りから正解に変わった録音が7件、正解から誤りに変わった録音が4件あった。予備比較に使った600件を除いても、GGUF CPUは5,296/5,400件から5,300/5,400件になった。

校正に使っていないFLEURSテスト分割でも、8言語の80録音をONNX CPUで確認した。元音声は79/80件から80/80件、先頭3秒は32/40件から33/40件、白色雑音を加えた音声（SNR 10 dB）は両モデルとも36/40件だった。合計は147/160件から149/160件で、基準版が正解して小型版が誤った例はこのテストにはなかった。

速度測定では上と同じPCで、モデルを読み込んで12件ウォームアップした後、同じ120件のMP3を各モデル2回ずつ処理した。基準版、小型版、小型版、基準版の順で実行し、各240回のデコード開始から判定までの所要時間を集計した。CPU推論は8スレッド、Pコア8個を使用した。

| 実装 | 基準版 中央値 | 小型版 中央値 | 基準版 p95 | 小型版 p95 |
|---|---:|---:|---:|---:|
| Rust GGUF CPU | 74.44 ms | 72.06 ms | 91.07 ms | 86.21 ms |
| Rust GGUF Vulkan | 11.36 ms | 11.59 ms | 28.71 ms | 27.34 ms |
| ONNX Runtime CPU（Python） | 91.18 ms | 84.60 ms | 101.94 ms | 96.32 ms |

### Compact版とランク96版の比較

Compact版はエンコーダー4層のMLPを各1,536から1,408チャネルに縮め、第2層の後で時間方向の隣接2位置を平均する。後半2層が扱う位置は1,500から750に減り、入力音声は従来どおり先頭30秒まで使える。パラメータ数は11,675,904から11,282,176へ、GGUFは17.58 MBから15.02 MBへ減った。[校正・量子化・推論の手順](docs/model-conversion.md#エンコーダーを圧縮したcompact版)

Emilia-YODASの既存6,000件を前回と同じ16 kHz PCMで比較した（入力SHA-256照合、99言語からの第1候補、各言語1,000件）。

| モデル | ドイツ語 | 英語 | フランス語 | 日本語 | 韓国語 | 中国語 | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ランク96版 | 991 | 993 | 988 | 970 | 991 | 955 | 5,888（98.13%） |
| Compact版（GGUF CPU・GGUF Vulkan・ONNX CPU） | 991 | 995 | 987 | 973 | 995 | 950 | 5,891（98.18%） |

3実装とも誤りから正解への変化が15件、正解から誤りへの変化が12件。GGUF CPUとONNX CPUの第1候補は全件一致した。Vulkanは誤判定した2件でCPUと異なる言語を選んだ。既存6,000件には候補の予備比較に使った600件を含む。モデル確定後に、元の6,000件と動画IDが重ならない別の600録音（各言語100件、1動画1録音、抽出シード741903）も比較した。GGUF Vulkanは592/600件から593/600件になった。

FLEURSの8言語・160条件では3実装とも、元音声80/80件、先頭3秒32/40件、白色雑音付加36/40件。ランク96版に比べて先頭3秒の正解が1件減り、合計は149/160件から148/160件になった。

CPU速度はCore i7-13700KFのPコア8個・8スレッドで測定した。各モデルの読み込み後に12件をウォームアップし、同じ120件のMP3を各2周、合計240回測定した。時間はファイルの読み込みから判定まで。RAMはWindowsのプロセスのピークワーキングセットで、2周の大きい方を示す。

| CPU実装 | ランク96版 中央値 → Compact版 | p95 | ピークRAM |
|---|---:|---:|---:|
| Rust GGUF | 76.01 → 55.44 ms | 91.67 → 70.19 ms | 105.2 → 99.4 MiB |
| ONNX Runtime（Python） | 89.62 → 56.11 ms | 103.18 → 66.32 ms | 571.4 → 365.4 MiB |

RTX 4070 TiのVulkanは測定順で差が出た。単独起動のCLIで同じ120件を各4周し、実行順を反転して合計480回を集計すると、中央値は11.25 → 10.79 ms、p95は25.48 → 23.24 msだった。先の2周と後の2周を分けると中央値はそれぞれ10.95 → 13.32 ms、11.45 → 8.55 ms。ホスト側のピークRAMは224.8 → 311.0 MiBだった（VRAMは測定対象外）。

両モデルを同時にロードし、同じMP3を1件ずつ交互に判定する比較も行った。1件ごと、周回ごとに先に実行するモデルを入れ替え、各4周480回で、読み込みから判定までの中央値は10.36 → 9.36 ms、PCMからの中央値は5.39 → 4.45 msだった。

#### 実時間係数（RTF）

同じ120件のMP3（判定対象の音声長は合計1,201.14秒）について、ファイル読み込みから判定までの合計処理時間を合計音声長で割った。RTFが小さいほど高速。単独起動の測定を使い、モデルのロードとウォームアップを除き、CPUは2周、Vulkanは4周を集計した。

| 実装 | ランク96版 RTF | Compact版 RTF | Compact版の120件当たりの平均処理時間 |
|---|---:|---:|---:|
| Rust GGUF・CPU | 0.00776 | 0.00564 | 6.77秒 |
| Rust GGUF・Vulkan | 0.00130 | 0.00120 | 1.44秒 |
| ONNX Runtime・CPU | 0.00911 | 0.00577 | 6.93秒 |

## ライセンス

このプロジェクトのコードは [MIT](LICENSE)、Whisper由来のモデルと前処理データには [OpenAIのMITライセンス](model/LICENSE) が適用されます。ggml・LLVM OpenMPの通知、音声デコーダーSymphoniaのMPL-2.0、Rust依存関係のライセンス一覧は [licenses/](licenses/) にあります。
