import Foundation
import Vision

do {
    guard CommandLine.arguments.count == 2 || CommandLine.arguments.count == 3 else { throw NSError(domain: "PhotoRecall", code: 1) }
    let handler = VNImageRequestHandler(url: URL(fileURLWithPath: CommandLine.arguments[1]), options: [:])
    if CommandLine.arguments.count == 3 && CommandLine.arguments[2] == "detect" {
        let detection = VNDetectTextRectanglesRequest()
        detection.reportCharacterBoxes = false
        try handler.perform([detection])
        let boxes = (detection.results ?? []).map { observation -> [Double] in
            let b = observation.boundingBox
            return [b.minX, 1-b.maxY, b.maxX, 1-b.minY]
        }
        let json = try JSONSerialization.data(withJSONObject: ["regions": boxes])
        print(String(data: json, encoding: .utf8)!)
        exit(0)
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    let supported = try request.supportedRecognitionLanguages()
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"].filter { supported.contains($0) }
    try handler.perform([request])
    let lines: [[String: Any]] = (request.results ?? []).compactMap { observation in
        guard let text = observation.topCandidates(1).first else { return nil }
        let b = observation.boundingBox
        return ["text": text.string, "confidence": text.confidence,
                "box": [b.minX, 1-b.maxY, b.maxX, 1-b.minY]]
    }
    let json = try JSONSerialization.data(withJSONObject: ["lines": lines])
    print(String(data: json, encoding: .utf8)!)
} catch {
    fputs("\(error)\n", stderr)
    exit(1)
}
