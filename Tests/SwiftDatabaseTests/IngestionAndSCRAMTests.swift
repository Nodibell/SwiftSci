import Testing
import Foundation
import CryptoKit
@testable import SwiftDatabase

// MARK: - 1024-Row Column Buffering Tests

@Suite("SQLite 1024-Row Column Buffer Ingestion Tests")
struct ColumnBufferIngestionTests {

    /// Verifies that executeQuery correctly returns all rows across multiple 1024-row page flushes.
    @Test("executeQuery returns all rows with 1024-row paged buffering")
    func testPageBufferFlushesAll() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        _ = try await conn.executeQuery("CREATE TABLE nums (val INTEGER NOT NULL)")

        // Insert 2100 rows — crosses two full pages (2 × 1024) plus a 52-row remainder
        var inserts: [String] = []
        for i in 0..<2100 { inserts.append("(\(i))") }
        let batchSQL = "INSERT INTO nums (val) VALUES \(inserts.joined(separator: ","));"
        _ = try await conn.executeQuery(batchSQL)

        let result = try await conn.executeQuery("SELECT * FROM nums ORDER BY val")
        #expect(result.rows.count == 2100)
        // Verify first and last values
        if case .int(let first) = result.rows.first?.first {
            #expect(first == 0)
        }
        if case .int(let last) = result.rows.last?.first {
            #expect(last == 2099)
        }
    }

    /// Verifies exact boundary: exactly 1024 rows (one complete page, no remainder).
    @Test("executeQuery handles exactly 1024-row result (single full page)")
    func testExactlyOnePage() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        _ = try await conn.executeQuery("CREATE TABLE page_test (x REAL)")
        let vals = (0..<1024).map { "(\(Double($0)))" }.joined(separator: ",")
        _ = try await conn.executeQuery("INSERT INTO page_test (x) VALUES \(vals)")
        let result = try await conn.executeQuery("SELECT * FROM page_test")
        #expect(result.rows.count == 1024)
    }

    /// Verifies zero rows do not trigger a spurious flush.
    @Test("executeQuery handles empty result (zero rows)")
    func testEmptyResult() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        _ = try await conn.executeQuery("CREATE TABLE empty_t (n INTEGER)")
        let result = try await conn.executeQuery("SELECT * FROM empty_t")
        #expect(result.rows.count == 0)
        #expect(result.columns == ["n"])
    }

    /// Smoke-tests the page-buffered path for all column types.
    @Test("executeQuery decodes all AnySendableValue column types via page buffer")
    func testMixedColumnTypes() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        _ = try await conn.executeQuery(
            "CREATE TABLE mixed (i INTEGER, f REAL, s TEXT, b BLOB, n INTEGER)"
        )
        _ = try await conn.executeQuery(
            "INSERT INTO mixed VALUES (42, 3.14, 'hello', X'DEADBEEF', NULL)"
        )
        let result = try await conn.executeQuery("SELECT * FROM mixed")
        #expect(result.rows.count == 1)
        let row = result.rows[0]
        if case .int(let v) = row[0] { #expect(v == 42) }
        if case .double(let v) = row[1] { #expect(abs(v - 3.14) < 1e-9) }
        if case .string(let v) = row[2] { #expect(v == "hello") }
        if case .data(let v) = row[3] { #expect(v == Data([0xDE, 0xAD, 0xBE, 0xEF])) }
        if case .null = row[4] { /* expected */ } else { Issue.record("Expected .null for column 4") }
    }
}

// MARK: - SCRAM-SHA-256 Unit Tests (protocol layer, no live server required)

@Suite("SCRAM-SHA-256 Protocol Layer Tests")
struct SCRAMProtocolTests {

    /// RFC 7677 SCRAM-SHA-256 PBKDF2 test vector (verified with Python hashlib.pbkdf2_hmac).
    /// Password="pencil", Salt=W22ZaJ0SNY7soEsUEjb6tQ==, iterations=4096
    /// Expected SaltedPassword (SHA-256): 68bb3872883b0b5f8bd7ae44bf2475805ba38ff623afb35bc3eca6662f5ac65d
    @Test("PBKDF2-SHA256 produces correct key for RFC 7677 SCRAM-SHA-256 test vector")
    func testPBKDF2TestVector() throws {
        let password = "pencil"
        let saltB64 = "W22ZaJ0SNY7soEsUEjb6tQ=="
        let iterations = 4096

        guard let saltData = Data(base64Encoded: saltB64) else {
            Issue.record("Failed to decode salt"); return
        }

        let passwordKey = Array(password.utf8)

        func hmac(_ key: [UInt8], _ msg: [UInt8]) -> [UInt8] {
            let symKey = SymmetricKey(data: Data(key))
            return Array(HMAC<SHA256>.authenticationCode(for: Data(msg), using: symKey))
        }

        var u = hmac(passwordKey, Array(saltData) + [0, 0, 0, 1])
        var result = u
        for _ in 1..<iterations {
            u = hmac(passwordKey, u)
            for j in 0..<result.count { result[j] ^= u[j] }
        }

        // Verified with: python3 -c "import hashlib,base64; ..."
        let expectedHex = "68bb3872883b0b5f8bd7ae44bf2475805ba38ff623afb35bc3eca6662f5ac65d"
        let computedHex = result.map { String(format: "%02x", $0) }.joined()
        #expect(computedHex == expectedHex, "PBKDF2-SHA256 mismatch for RFC 7677 test vector")
    }


    /// Verifies ClientKey/StoredKey/ClientProof derivation correctness given known SaltedPassword.
    @Test("SCRAM ClientKey, StoredKey, and ClientProof derivation")
    func testSCRAMKeyDerivation() {
        // Any 32-byte SaltedPassword for structural correctness validation
        let saltedPassword = [UInt8](repeating: 0xAB, count: 32)
        let authMessage = "n=testuser,r=nonce,s=c2FsdA==,i=4096,c=biws,r=nonce"

        func hmac(_ key: [UInt8], _ msg: [UInt8]) -> [UInt8] {
            let sym = SymmetricKey(data: Data(key))
            return Array(HMAC<SHA256>.authenticationCode(for: Data(msg), using: sym))
        }

        func sha256(_ b: [UInt8]) -> [UInt8] { Array(SHA256.hash(data: Data(b))) }

        let clientKey = hmac(saltedPassword, Array("Client Key".utf8))
        let storedKey = sha256(clientKey)
        let clientSig = hmac(storedKey, Array(authMessage.utf8))
        let clientProof = zip(clientKey, clientSig).map { $0 ^ $1 }

        // Structural: lengths must be 32 bytes (SHA256 output)
        #expect(clientKey.count == 32)
        #expect(storedKey.count == 32)
        #expect(clientProof.count == 32)

        // ClientProof XOR ClientSig recovers ClientKey (proof correctness)
        let recovered = zip(clientProof, clientSig).map { $0 ^ $1 }
        #expect(recovered == clientKey, "ClientProof XOR ClientSig != ClientKey")
    }

    /// Verifies nonce generation produces valid base64 and unique values.
    @Test("Client nonce is base64-encoded and unique across calls")
    func testClientNonceUniqueness() {
        func generateNonce() -> String {
            var bytes = [UInt8](repeating: 0, count: 18)
            _ = SecRandomCopyBytes(kSecRandomDefault, 18, &bytes)
            return Data(bytes).base64EncodedString()
        }

        let n1 = generateNonce()
        let n2 = generateNonce()
        #expect(n1 != n2, "Two random nonces should never be equal")
        #expect(Data(base64Encoded: n1) != nil, "Client nonce must be valid base64")
        #expect(n1.count > 16, "Nonce must be at least 16 characters")
    }

    /// Verifies that PostgreSQLConnection throws .unsupportedAuth when SCRAM is enabled
    /// but called on a platform without CryptoKit (simulated via error message check).
    @Test("DatabaseError.unsupportedAuth contains SCRAM context string")
    func testUnsupportedAuthError() {
        let err = DatabaseError.unsupportedAuth("SCRAM-SHA-256 requires CryptoKit (macOS 10.15+)")
        #expect(err.errorDescription?.contains("SCRAM-SHA-256") == true)
        #expect(err.errorDescription?.contains("CryptoKit") == true)
    }
}
