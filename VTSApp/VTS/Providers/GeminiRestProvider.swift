import Foundation

public class GeminiRestProvider: BaseRestSTTProvider {
    public override var providerType: STTProviderType { .gemini }

    private let baseURL = "https://generativelanguage.googleapis.com/v1beta/models"

    public override init() {
        super.init()
    }

    public override func transcribe(
        stream: AsyncThrowingStream<Data, Error>,
        config: ProviderConfig
    ) async throws -> String {
        var audioData = Data()
        print("Gemini: Starting audio collection...")

        for try await chunk in stream {
            audioData.append(chunk)
            print("Gemini: Received audio chunk of \(chunk.count) bytes, total: \(audioData.count)")
        }

        print("Gemini: Audio collection completed, total size: \(audioData.count) bytes")

        let minimumBytes = Int(24000 * 1)
        guard audioData.count >= minimumBytes else {
            print("Gemini: Not enough audio data (\(audioData.count) bytes, minimum: \(minimumBytes))")
            throw STTError.audioProcessingError("Not enough audio data")
        }

        print("Gemini: Sending transcription request...")
        let result = try await sendTranscriptionRequest(audioData: audioData, config: config)
        print("Gemini: Received transcription result: \(result)")

        return result
    }

    public override func validateConfig(_ config: ProviderConfig) throws {
        guard !config.apiKey.isEmpty else {
            throw STTError.invalidAPIKey
        }

        guard STTProviderType.gemini.restModels.contains(config.model) else {
            throw STTError.invalidModel
        }
    }

    private func sendTranscriptionRequest(audioData: Data, config: ProviderConfig) async throws -> String {
        let urlString = "\(baseURL)/\(config.model):generateContent?key=\(config.apiKey)"
        var request = URLRequest(url: URL(string: urlString)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let wavData = createWAVData(from: audioData)
        let base64Audio = wavData.base64EncodedString()

        var promptText = "Please transcribe the audio accurately. Return only the transcribed text with no additional commentary."
        if let systemPrompt = config.systemPrompt, !systemPrompt.isEmpty {
            promptText = "\(systemPrompt)\n\nPlease transcribe the audio accurately. Return only the transcribed text with no additional commentary."
        }

        let body: [String: Any] = [
            "contents": [
                [
                    "parts": [
                        [
                            "inline_data": [
                                "mime_type": "audio/wav",
                                "data": base64Audio
                            ]
                        ],
                        [
                            "text": promptText
                        ]
                    ]
                ]
            ]
        ]

        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await performNetworkRequest(
            request: request,
            audioDataSize: audioData.count,
            providerName: "Gemini"
        )

        guard let httpResponse = response as? HTTPURLResponse else {
            throw STTError.networkError("Invalid response format")
        }

        switch httpResponse.statusCode {
        case 200...299:
            break

        case 400:
            let errorDetails = parseErrorResponse(data)
            if errorDetails.lowercased().contains("api key") {
                throw STTError.invalidAPIKey
            } else {
                throw STTError.transcriptionError("Bad request: \(errorDetails)")
            }

        case 401, 403:
            throw STTError.invalidAPIKey

        case 429:
            throw STTError.transcriptionError("Rate limit exceeded. Please wait before trying again.")

        case 500...599:
            let errorDetails = parseErrorResponse(data)
            throw STTError.networkError("Server error (\(httpResponse.statusCode)): \(errorDetails)")

        default:
            let errorDetails = parseErrorResponse(data)
            throw STTError.networkError("Request failed (\(httpResponse.statusCode)): \(errorDetails)")
        }

        struct GeminiResponse: Codable {
            let candidates: [Candidate]

            struct Candidate: Codable {
                let content: Content

                struct Content: Codable {
                    let parts: [Part]

                    struct Part: Codable {
                        let text: String
                    }
                }
            }
        }

        let geminiResponse = try JSONDecoder().decode(GeminiResponse.self, from: data)
        guard let text = geminiResponse.candidates.first?.content.parts.first?.text else {
            throw STTError.transcriptionError("Empty transcription response from Gemini")
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func parseErrorResponse(_ data: Data) -> String {
        struct GeminiErrorResponse: Codable {
            let error: ErrorDetails

            struct ErrorDetails: Codable {
                let message: String
            }
        }

        if let errorResponse = try? JSONDecoder().decode(GeminiErrorResponse.self, from: data) {
            return errorResponse.error.message
        }

        if let responseText = String(data: data, encoding: .utf8), !responseText.isEmpty {
            return responseText
        }

        return "Unknown error"
    }
}
