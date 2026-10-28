import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftDatabase

@Suite("Wire Protocol Driver Mock Server Tests")
struct WireProtocolMockTests {

    /// Helper to start a local TCP mock server on an ephemeral port.
    private static func createMockServer(handler: @escaping @Sendable (Int32) -> Void) -> (port: Int, stop: () -> Void) {
        let serverFd = socket(AF_INET, SOCK_STREAM, 0)
        var opt: Int32 = 1
        setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0 // Ephemeral port
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                _ = bind(serverFd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        listen(serverFd, 5)

        var assignedAddr = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        withUnsafeMutablePointer(to: &assignedAddr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                _ = getsockname(serverFd, $0, &len)
            }
        }
        let port = Int(UInt16(bigEndian: assignedAddr.sin_port))

        let serverReady = DispatchSemaphore(value: 0)

        DispatchQueue.global(qos: .userInitiated).async {
            serverReady.signal()
            var clientAddr = sockaddr_in()
            var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
            let clientFd = withUnsafeMutablePointer(to: &clientAddr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    accept(serverFd, $0, &clientLen)
                }
            }
            if clientFd >= 0 {
                var tv = timeval(tv_sec: 5, tv_usec: 0)
                setsockopt(clientFd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
                setsockopt(clientFd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

                handler(clientFd)

                // Gracefully wait for client to read and finish
                shutdown(clientFd, SHUT_WR)
                var discard = [UInt8](repeating: 0, count: 256)
                while recv(clientFd, &discard, discard.count, 0) > 0 {}
                close(clientFd)
            }
        }

        serverReady.wait()
        // Brief yield to ensure OS kernel has accept thread scheduled
        Thread.sleep(forTimeInterval: 0.01)

        let stop: () -> Void = {
            _ = close(serverFd)
        }

        return (port, stop)
    }

    @Test("PostgreSQL trust auth and query response wire decoding")
    func testPostgresTrustAuthAndQuery() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)
            _ = recv(clientFd, &buf, buf.count, 0)

            // AuthOK (type 0) + ReadyForQuery ('Z')
            var authOk = Data([0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00])
            authOk.append(contentsOf: [0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = authOk.withUnsafeBytes { send(clientFd, $0.baseAddress, authOk.count, 0) }

            // Read query packet 'Q'
            _ = recv(clientFd, &buf, buf.count, 0)

            var resp = Data()
            var tBody = Data([0x00, 0x02])
            tBody.append(contentsOf: "id\0".utf8)
            tBody.append(contentsOf: [UInt8](repeating: 0, count: 18))
            tBody.append(contentsOf: "name\0".utf8)
            tBody.append(contentsOf: [UInt8](repeating: 0, count: 18))

            resp.append(0x54)
            withUnsafeBytes(of: Int32(tBody.count + 4).bigEndian) { resp.append(contentsOf: $0) }
            resp.append(tBody)

            var dBody = Data([0x00, 0x02])
            let val1 = "1"
            withUnsafeBytes(of: Int32(val1.utf8.count).bigEndian) { dBody.append(contentsOf: $0) }
            dBody.append(contentsOf: val1.utf8)
            let val2 = "Alice"
            withUnsafeBytes(of: Int32(val2.utf8.count).bigEndian) { dBody.append(contentsOf: $0) }
            dBody.append(contentsOf: val2.utf8)

            resp.append(0x44)
            withUnsafeBytes(of: Int32(dBody.count + 4).bigEndian) { resp.append(contentsOf: $0) }
            resp.append(dBody)

            resp.append(contentsOf: [0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = resp.withUnsafeBytes { send(clientFd, $0.baseAddress, resp.count, 0) }
        }
        defer { stop() }

        let conn = PostgreSQLConnection(connectionURL: "postgres://testuser@127.0.0.1:\(port)/testdb")
        let result = try await conn.executeQuery("SELECT id, name FROM users;")
        #expect(result.columns == ["id", "name"])
        #expect(result.rows.count == 1)
        #expect(result.rows[0][0].description == "1")
        #expect(result.rows[0][1].description == "Alice")
    }

    @Test("PostgreSQL MD5 auth handshake flow")
    func testPostgresMD5Auth() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)
            _ = recv(clientFd, &buf, buf.count, 0)

            // Send MD5 challenge: 'R', len(12), type(5), 4-byte salt [1, 2, 3, 4]
            let md5Req = Data([0x52, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x05, 0x01, 0x02, 0x03, 0x04])
            _ = md5Req.withUnsafeBytes { send(clientFd, $0.baseAddress, md5Req.count, 0) }

            // Read password response 'p'
            _ = recv(clientFd, &buf, buf.count, 0)

            // Send AuthenticationOk + ReadyForQuery
            var authOk = Data([0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00])
            authOk.append(contentsOf: [0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = authOk.withUnsafeBytes { send(clientFd, $0.baseAddress, authOk.count, 0) }

            // Read Query
            _ = recv(clientFd, &buf, buf.count, 0)

            // Send ReadyForQuery for the query
            let rfq = Data([0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = rfq.withUnsafeBytes { send(clientFd, $0.baseAddress, rfq.count, 0) }
        }
        defer { stop() }

        let conn = PostgreSQLConnection(connectionURL: "postgres://user:password@127.0.0.1:\(port)/testdb")
        let res = try await conn.executeQuery("SELECT 1;")
        #expect(res.rows.isEmpty)
    }

    @Test("PostgreSQL unsupported auth error response")
    func testPostgresUnsupportedAuth() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)
            _ = recv(clientFd, &buf, buf.count, 0)

            // Send auth type 10 (SASL / SCRAM)
            let scramReq = Data([0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x0A])
            _ = scramReq.withUnsafeBytes { send(clientFd, $0.baseAddress, scramReq.count, 0) }
        }
        defer { stop() }

        let conn = PostgreSQLConnection(connectionURL: "postgres://user:pass@127.0.0.1:\(port)/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("SELECT 1;")
        }
    }

    @Test("PostgreSQL server error response packet ('E')")
    func testPostgresErrorResponse() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)
            _ = recv(clientFd, &buf, buf.count, 0)

            // Send 'E' ErrorResponse
            var errResp = Data([0x45])
            let errMsg = "SERROR\0C42P01\0Mrelation does not exist\0\0"
            withUnsafeBytes(of: Int32(errMsg.utf8.count + 4).bigEndian) { errResp.append(contentsOf: $0) }
            errResp.append(contentsOf: errMsg.utf8)
            _ = errResp.withUnsafeBytes { send(clientFd, $0.baseAddress, errResp.count, 0) }
        }
        defer { stop() }

        let conn = PostgreSQLConnection(connectionURL: "postgres://user:pass@127.0.0.1:\(port)/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("SELECT * FROM invalid_table;")
        }
    }

    @Test("MySQL wire handshake and query response wire decoding")
    func testMySQLHandshakeAndQuery() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)

            // 1. Send Handshake packet (v10)
            var handshakePayload = Data([10])
            handshakePayload.append(contentsOf: "8.0.32\0".utf8)
            handshakePayload.append(contentsOf: [1, 0, 0, 0])
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x41, count: 8))
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: [0xFF, 0xF7])
            handshakePayload.append(33)
            handshakePayload.append(contentsOf: [0x02, 0x00])
            handshakePayload.append(contentsOf: [0xFF, 0x81])
            handshakePayload.append(21)
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x00, count: 10))
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x42, count: 12))
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: "mysql_native_password\0".utf8)

            let hLen = handshakePayload.count
            var handshakePacket = Data([UInt8(hLen & 0xFF), UInt8((hLen >> 8) & 0xFF), UInt8((hLen >> 16) & 0xFF), 0])
            handshakePacket.append(handshakePayload)
            _ = handshakePacket.withUnsafeBytes { send(clientFd, $0.baseAddress, handshakePacket.count, 0) }

            // 2. Read Auth Response
            _ = recv(clientFd, &buf, buf.count, 0)

            // 3. Send Auth OK packet
            let okPayload = Data([0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00])
            let okLen = okPayload.count
            var okPacket = Data([UInt8(okLen & 0xFF), UInt8((okLen >> 8) & 0xFF), UInt8((okLen >> 16) & 0xFF), 2])
            okPacket.append(okPayload)
            _ = okPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, okPacket.count, 0) }

            // 4. Read COM_QUERY
            _ = recv(clientFd, &buf, buf.count, 0)

            // 5. Send Result Set: 1 column ("score"), 1 row (99.5)
            let colCountPacket = Data([1, 0, 0, 1, 1])
            _ = colCountPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, colCountPacket.count, 0) }

            var colDefPayload = Data()
            func appendLenc(_ s: String) {
                colDefPayload.append(UInt8(s.utf8.count))
                colDefPayload.append(contentsOf: s.utf8)
            }
            appendLenc("def")
            appendLenc("testdb")
            appendLenc("orders")
            appendLenc("orders")
            appendLenc("score")
            appendLenc("score")
            colDefPayload.append(contentsOf: [UInt8](repeating: 0, count: 12))

            let cdLen = colDefPayload.count
            var colDefPacket = Data([UInt8(cdLen & 0xFF), UInt8((cdLen >> 8) & 0xFF), UInt8((cdLen >> 16) & 0xFF), 2])
            colDefPacket.append(colDefPayload)
            _ = colDefPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, colDefPacket.count, 0) }

            let eofPacket = Data([5, 0, 0, 3, 0xFE, 0x00, 0x00, 0x02, 0x00])
            _ = eofPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, eofPacket.count, 0) }

            var rowPayload = Data()
            rowPayload.append(UInt8("99.5".utf8.count))
            rowPayload.append(contentsOf: "99.5".utf8)
            let rLen = rowPayload.count
            var rowPacket = Data([UInt8(rLen & 0xFF), UInt8((rLen >> 8) & 0xFF), UInt8((rLen >> 16) & 0xFF), 4])
            rowPacket.append(rowPayload)
            _ = rowPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, rowPacket.count, 0) }

            let eofPacket2 = Data([5, 0, 0, 5, 0xFE, 0x00, 0x00, 0x02, 0x00])
            _ = eofPacket2.withUnsafeBytes { send(clientFd, $0.baseAddress, eofPacket2.count, 0) }
        }
        defer { stop() }

        let conn = MySQLConnection(connectionURL: "mysql://root:secret@127.0.0.1:\(port)/testdb")
        let result = try await conn.executeQuery("SELECT score FROM orders;")
        #expect(result.columns == ["score"])
        #expect(result.rows.count == 1)
        #expect(result.rows[0][0].description == "99.5")
    }

    @Test("MySQL query error packet and OK DDL packet")
    func testMySQLErrAndOKPacket() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)

            // 1. Handshake
            var handshakePayload = Data([10])
            handshakePayload.append(contentsOf: "8.0.32\0".utf8)
            handshakePayload.append(contentsOf: [1, 0, 0, 0])
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x41, count: 8))
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: [0xFF, 0xF7])
            handshakePayload.append(33)
            handshakePayload.append(contentsOf: [0x02, 0x00])
            handshakePayload.append(contentsOf: [0xFF, 0x81])
            handshakePayload.append(21)
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x00, count: 10))
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x42, count: 12))
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: "mysql_native_password\0".utf8)

            let hLen = handshakePayload.count
            var handshakePacket = Data([UInt8(hLen & 0xFF), UInt8((hLen >> 8) & 0xFF), UInt8((hLen >> 16) & 0xFF), 0])
            handshakePacket.append(handshakePayload)
            _ = handshakePacket.withUnsafeBytes { send(clientFd, $0.baseAddress, handshakePacket.count, 0) }

            // 2. Read Auth
            _ = recv(clientFd, &buf, buf.count, 0)

            // 3. Send Auth OK
            let okPayload = Data([0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00])
            let okLen = okPayload.count
            var okPacket = Data([UInt8(okLen & 0xFF), UInt8((okLen >> 8) & 0xFF), UInt8((okLen >> 16) & 0xFF), 2])
            okPacket.append(okPayload)
            _ = okPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, okPacket.count, 0) }

            // 4. Read COM_QUERY
            _ = recv(clientFd, &buf, buf.count, 0)

            // 5. Send OK packet for DDL query (first byte 0x00)
            let ddlOkPacket = Data([7, 0, 0, 1, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00])
            _ = ddlOkPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, ddlOkPacket.count, 0) }
        }
        defer { stop() }

        let conn = MySQLConnection(connectionURL: "mysql://root:secret@127.0.0.1:\(port)/testdb")
        let res = try await conn.executeQuery("CREATE TABLE t (x INT);")
        #expect(res.rows.isEmpty)
    }

    @Test("PostgreSQL SCRAM-SHA-256 complete wire authentication flow")
    func testPostgresSCRAMAuthSuccess() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)
            // 1. Read StartupMessage
            _ = recv(clientFd, &buf, buf.count, 0)

            // 2. Send AuthenticationSASL (type 10): 'R', len(23), type 10, "SCRAM-SHA-256\0\0"
            var saslReq = Data([0x52])
            let mech = "SCRAM-SHA-256\0\0"
            let saslLen = Int32(8 + mech.utf8.count).bigEndian
            withUnsafeBytes(of: saslLen) { saslReq.append(contentsOf: $0) }
            saslReq.append(contentsOf: [0x00, 0x00, 0x00, 0x0A]) // AuthType 10
            saslReq.append(contentsOf: mech.utf8)
            _ = saslReq.withUnsafeBytes { send(clientFd, $0.baseAddress, saslReq.count, 0) }

            // 3. Read SASLInitialResponse 'p'
            let rLen = recv(clientFd, &buf, buf.count, 0)
            guard rLen > 0 else { return }
            let clientInitStr = String(decoding: buf[0..<Int(rLen)], as: UTF8.self)

            // Extract client nonce from n,,n=user,r=<clientNonce>
            var clientNonce = "mockNonce"
            if let rRange = clientInitStr.range(of: ",r=") {
                let afterR = clientInitStr[rRange.upperBound...]
                let nonceCandidate = afterR.prefix(while: { $0 != "," && $0 != "\0" })
                if !nonceCandidate.isEmpty { clientNonce = String(nonceCandidate) }
            }

            // 4. Send AuthenticationSASLContinue (type 11): 'R', len, type 11, payload "r=...,s=...,i=4096"
            let serverNonce = clientNonce + "SERVER_NONCE_EXT"
            let saltB64 = "W22ZaJ0SNY7soEsUEjb6tQ=="
            let serverFirstPayload = "r=\(serverNonce),s=\(saltB64),i=4096"

            var contPacket = Data([0x52])
            let contLen = Int32(8 + serverFirstPayload.utf8.count).bigEndian
            withUnsafeBytes(of: contLen) { contPacket.append(contentsOf: $0) }
            contPacket.append(contentsOf: [0x00, 0x00, 0x00, 0x0B]) // AuthType 11
            contPacket.append(contentsOf: serverFirstPayload.utf8)
            _ = contPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, contPacket.count, 0) }

            // 5. Read SASLResponse 'p'
            _ = recv(clientFd, &buf, buf.count, 0)

            // 6. Send AuthenticationOk (type 0) + ReadyForQuery
            var authOk = Data([0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00])
            authOk.append(contentsOf: [0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = authOk.withUnsafeBytes { send(clientFd, $0.baseAddress, authOk.count, 0) }

            // 7. Read Query packet
            _ = recv(clientFd, &buf, buf.count, 0)

            // 8. Send ReadyForQuery
            let rfq = Data([0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = rfq.withUnsafeBytes { send(clientFd, $0.baseAddress, rfq.count, 0) }
        }
        defer { stop() }

        let conn = PostgreSQLConnection(connectionURL: "postgres://scramuser:pencil@127.0.0.1:\(port)/testdb")
        let res = try await conn.executeQuery("SELECT 1;")
        #expect(res.rows.isEmpty)
    }

    @Test("SQLQueryResult and DataFrame.fromSQL type marshalling for all extended types")
    func testQueryResultToDataFrameExtendedTypes() async throws {
        let testDate = Date(timeIntervalSince1970: 1700000000)
        let testData = Data([0xAA, 0xBB, 0xCC])

        // 1. Test SQLQueryResult.toDataFrame() with all types:
        let qResult = SQLQueryResult(
            columns: ["col_i64", "col_bool", "col_date", "col_data", "col_str", "col_dbl", "col_null"],
            rows: [
                [.int64(999), .bool(true), .date(testDate), .data(testData), .string("test"), .double(3.14), .null],
                [.int64(1000), .bool(false), .date(testDate), .data(testData), .string("test2"), .double(6.28), .null]
            ]
        )

        let df = try qResult.toDataFrame()
        #expect(df.shape.rows == 2)
        #expect(df.shape.columns == 7)

        // 2. Test DataFrame.fromSQL with mock connection returning extended types
        struct MockConnection: DatabaseConnection {
            let result: SQLQueryResult
            func executeQuery(_ query: String) async throws -> SQLQueryResult {
                result
            }
        }

        let mockConn = MockConnection(result: qResult)
        let dfFromSQL = try await DataFrame.fromSQL("SELECT * FROM mock;", connection: mockConn)
        #expect(dfFromSQL.shape.rows == 2)
        #expect(dfFromSQL.shape.columns == 7)
    }

    @Test("PostgreSQL and MySQL fail closed when sslMode is .require")
    func testTLSRequireFailsClosed() async {
        let pg = PostgreSQLConnection(
            host: "127.0.0.1",
            port: 5432,
            user: "postgres",
            password: "secret",
            database: "testdb",
            sslMode: .require
        )
        await #expect(throws: DatabaseError.self) {
            _ = try await pg.executeQuery("SELECT 1;")
        }

        let my = MySQLConnection(
            host: "127.0.0.1",
            port: 3306,
            user: "root",
            password: "secret",
            database: "testdb",
            sslMode: .require
        )
        await #expect(throws: DatabaseError.self) {
            _ = try await my.executeQuery("SELECT 1;")
        }
    }

    @Test("MySQL sends 20-byte SHA-1 token matching reference vector, decodes HandshakeResponse41 with database, and rejects unsupported auth")
    func testMySQLNativePasswordTokenAndUnsupportedPlugin() async throws {
        // 1. Verify against independent Python hashlib.sha1 reference vector:
        // password = "secret", scramble = "12345678901234567890"
        // SHA1("secret") = e5e9fa1ba31ecd1ae84f75caaa474f3a663f05f4
        // SHA1(SHA1("secret")) = 81f5e21e35407d884a6cd4a731aebfb6af209e1b
        // SHA1(scramble + stage2) = ea626a284397b1106b779e2974ee4e37bc78aea2
        // stage1 XOR stage3 = 0f8b9033e0897c0a8338ebe3dea9010dda47ab56
        let refScramble = Array("12345678901234567890".utf8)
        let refToken = MySQLConnection.mysqlNativePasswordToken(
            password: "secret",
            scramble: refScramble
        )
        let expectedRefBytes: [UInt8] = [
            0x0F, 0x8B, 0x90, 0x33, 0xE0, 0x89, 0x7C, 0x0A, 0x83, 0x38,
            0xEB, 0xE3, 0xDE, 0xA9, 0x01, 0x0D, 0xDA, 0x47, 0xAB, 0x56
        ]
        #expect(refToken == expectedRefBytes)

        // 2. Verify full HandshakeResponse41 decoding with non-empty database ("analytics_db")
        let scramble1 = Array("12345678".utf8)
        let scramble2 = Array("901234567890".utf8)
        final class HandshakeCapture: @unchecked Sendable {
            var clientFlags: UInt32 = 0
            var username: String = ""
            var authToken: [UInt8] = []
            var database: String = ""
            var pluginName: String = ""
        }
        let capture = HandshakeCapture()

        let (validPort, stopValid) = Self.createMockServer { clientFd in
            // Send HandshakeV10 with mysql_native_password
            var handshakePayload = Data([10])
            handshakePayload.append(contentsOf: "8.0.32\0".utf8)
            handshakePayload.append(contentsOf: [1, 0, 0, 0])
            handshakePayload.append(contentsOf: scramble1)
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: [0xFF, 0xF7])
            handshakePayload.append(33)
            handshakePayload.append(contentsOf: [0x02, 0x00])
            handshakePayload.append(contentsOf: [0xFF, 0x81])
            handshakePayload.append(21)
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x00, count: 10))
            handshakePayload.append(contentsOf: scramble2)
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: "mysql_native_password\0".utf8)

            let hLen = handshakePayload.count
            var handshakePacket = Data([UInt8(hLen & 0xFF), UInt8((hLen >> 8) & 0xFF), UInt8((hLen >> 16) & 0xFF), 0])
            handshakePacket.append(handshakePayload)
            _ = handshakePacket.withUnsafeBytes { send(clientFd, $0.baseAddress, handshakePacket.count, 0) }

            // Read client HandshakeResponse41 packet header + payload
            var hdr = [UInt8](repeating: 0, count: 4)
            guard recv(clientFd, &hdr, 4, MSG_WAITALL) == 4 else { return }
            let pLen = Int(hdr[0]) | (Int(hdr[1]) << 8) | (Int(hdr[2]) << 16)
            var payload = [UInt8](repeating: 0, count: pLen)
            guard recv(clientFd, &payload, pLen, MSG_WAITALL) == pLen, pLen >= 32 else { return }

            let flags = UInt32(payload[0])
                | (UInt32(payload[1]) << 8)
                | (UInt32(payload[2]) << 16)
                | (UInt32(payload[3]) << 24)
            capture.clientFlags = flags

            var idx = 32 // 4 (flags) + 4 (max_packet) + 1 (charset) + 23 (filler)
            var userBytes: [UInt8] = []
            while idx < payload.count && payload[idx] != 0 {
                userBytes.append(payload[idx])
                idx += 1
            }
            idx += 1 // skip null terminator
            capture.username = String(decoding: userBytes, as: UTF8.self)

            if idx < payload.count {
                let authLen = Int(payload[idx])
                idx += 1
                if idx + authLen <= payload.count {
                    capture.authToken = Array(payload[idx..<(idx + authLen)])
                    idx += authLen
                }
            }

            if (flags & 0x00000008) != 0 { // CLIENT_CONNECT_WITH_DB
                var dbBytes: [UInt8] = []
                while idx < payload.count && payload[idx] != 0 {
                    dbBytes.append(payload[idx])
                    idx += 1
                }
                idx += 1
                capture.database = String(decoding: dbBytes, as: UTF8.self)
            }

            if (flags & 0x00080000) != 0 { // CLIENT_PLUGIN_AUTH
                var plugBytes: [UInt8] = []
                while idx < payload.count && payload[idx] != 0 {
                    plugBytes.append(payload[idx])
                    idx += 1
                }
                idx += 1
                capture.pluginName = String(decoding: plugBytes, as: UTF8.self)
            }

            // Send Auth OK (0x00)
            let okAuth: [UInt8] = [0x07, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]
            _ = send(clientFd, okAuth, okAuth.count, 0)

            // Read COM_QUERY and reply with OK packet
            var qHdr = [UInt8](repeating: 0, count: 4)
            if recv(clientFd, &qHdr, 4, MSG_WAITALL) == 4 {
                let qLen = Int(qHdr[0]) | (Int(qHdr[1]) << 8) | (Int(qHdr[2]) << 16)
                var qPayload = [UInt8](repeating: 0, count: qLen)
                _ = recv(clientFd, &qPayload, qLen, MSG_WAITALL)
                let okQuery: [UInt8] = [0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]
                _ = send(clientFd, okQuery, okQuery.count, 0)
            }
        }
        defer { stopValid() }

        let validConn = MySQLConnection(connectionURL: "mysql://root:secret@127.0.0.1:\(validPort)/analytics_db")
        _ = try await validConn.executeQuery("SELECT 1;")
        #expect((capture.clientFlags & 0x00000008) != 0) // CLIENT_CONNECT_WITH_DB
        #expect((capture.clientFlags & 0x00008000) != 0) // CLIENT_SECURE_CONNECTION
        #expect((capture.clientFlags & 0x00080000) != 0) // CLIENT_PLUGIN_AUTH
        #expect(capture.username == "root")
        #expect(capture.authToken == expectedRefBytes)
        #expect(capture.database == "analytics_db")
        #expect(capture.pluginName == "mysql_native_password")

        // 3. Reject unsupported plugin in initial HandshakeV10
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)

            var handshakePayload = Data([10])
            handshakePayload.append(contentsOf: "8.0.32\0".utf8)
            handshakePayload.append(contentsOf: [1, 0, 0, 0])
            handshakePayload.append(contentsOf: scramble1)
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: [0xFF, 0xF7])
            handshakePayload.append(33)
            handshakePayload.append(contentsOf: [0x02, 0x00])
            handshakePayload.append(contentsOf: [0xFF, 0x81])
            handshakePayload.append(21)
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x00, count: 10))
            handshakePayload.append(contentsOf: scramble2)
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: "caching_sha2_password\0".utf8)

            let hLen = handshakePayload.count
            var handshakePacket = Data([UInt8(hLen & 0xFF), UInt8((hLen >> 8) & 0xFF), UInt8((hLen >> 16) & 0xFF), 0])
            handshakePacket.append(handshakePayload)
            _ = handshakePacket.withUnsafeBytes { send(clientFd, $0.baseAddress, handshakePacket.count, 0) }
            _ = recv(clientFd, &buf, buf.count, 0)
        }
        defer { stop() }

        let conn = MySQLConnection(connectionURL: "mysql://root:secret@127.0.0.1:\(port)/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("SELECT 1;")
        }

        // 4. Reject server AuthSwitchRequest (0xFE) after HandshakeResponse41 without sending COM_QUERY
        final class SwitchProbe: @unchecked Sendable {
            var receivedComQueryAfterSwitch = false
        }
        let switchProbe = SwitchProbe()
        let (switchPort, stopSwitch) = Self.createMockServer { clientFd in
            var handshakePayload = Data([10])
            handshakePayload.append(contentsOf: "8.0.32\0".utf8)
            handshakePayload.append(contentsOf: [1, 0, 0, 0])
            handshakePayload.append(contentsOf: scramble1)
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: [0xFF, 0xF7])
            handshakePayload.append(33)
            handshakePayload.append(contentsOf: [0x02, 0x00])
            handshakePayload.append(contentsOf: [0xFF, 0x81])
            handshakePayload.append(21)
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x00, count: 10))
            handshakePayload.append(contentsOf: scramble2)
            handshakePayload.append(0x00)
            handshakePayload.append(contentsOf: "mysql_native_password\0".utf8)

            let hLen = handshakePayload.count
            var handshakePacket = Data([UInt8(hLen & 0xFF), UInt8((hLen >> 8) & 0xFF), UInt8((hLen >> 16) & 0xFF), 0])
            handshakePacket.append(handshakePayload)
            _ = handshakePacket.withUnsafeBytes { send(clientFd, $0.baseAddress, handshakePacket.count, 0) }

            // Read client HandshakeResponse41
            var hdr = [UInt8](repeating: 0, count: 4)
            guard recv(clientFd, &hdr, 4, MSG_WAITALL) == 4 else { return }
            let pLen = Int(hdr[0]) | (Int(hdr[1]) << 8) | (Int(hdr[2]) << 16)
            var payload = [UInt8](repeating: 0, count: pLen)
            _ = recv(clientFd, &payload, pLen, MSG_WAITALL)

            // Reply with AuthSwitchRequest (0xFE)
            var switchPayload = Data([0xFE])
            switchPayload.append(contentsOf: "caching_sha2_password\0".utf8)
            switchPayload.append(contentsOf: scramble1 + scramble2)
            switchPayload.append(0x00)
            let sLen = switchPayload.count
            var switchPkt = Data([UInt8(sLen & 0xFF), UInt8((sLen >> 8) & 0xFF), UInt8((sLen >> 16) & 0xFF), 2])
            switchPkt.append(switchPayload)
            _ = switchPkt.withUnsafeBytes { send(clientFd, $0.baseAddress, switchPkt.count, 0) }

            // Ensure client disconnects rather than sending COM_QUERY (0x03)
            var nextBuf = [UInt8](repeating: 0, count: 16)
            let n = recv(clientFd, &nextBuf, nextBuf.count, 0)
            if n > 4 && nextBuf[4] == 0x03 {
                switchProbe.receivedComQueryAfterSwitch = true
            }
        }
        defer { stopSwitch() }

        let switchConn = MySQLConnection(connectionURL: "mysql://root:secret@127.0.0.1:\(switchPort)/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await switchConn.executeQuery("SELECT 1;")
        }
        #expect(switchProbe.receivedComQueryAfterSwitch == false)
    }

    @Test("SQLiteConnection executeReadOnlyQuery enforces C-API read-only and single-statement semantics")
    func testSQLiteExecuteReadOnlyQuery() async throws {
        let tempFile = NSTemporaryDirectory() + "test_ro_\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempFile) }

        // Verify same SQLiteConnection instance uses dedicated read-only handle even after read-write executeQuery
        let writer = SQLiteConnection(databasePath: tempFile)
        _ = try await writer.executeQuery("CREATE TABLE events (id INT PRIMARY KEY, created_at TEXT);")
        _ = try await writer.executeQuery("INSERT INTO events VALUES (1, '2026-10-28');")
        let sameConnReadOnlyResult = try await writer.executeReadOnlyQuery("SELECT id, created_at FROM events;")
        #expect(sameConnReadOnlyResult.rows.count == 1)
        #expect(try await writer.isHandleReadOnly(forReadOnlyQuery: false) == false)
        #expect(try await writer.isHandleReadOnly(forReadOnlyQuery: true) == true)
        await writer.close()

        let reader = SQLiteConnection(databasePath: tempFile)
        let res = try await reader.executeReadOnlyQuery("SELECT id, created_at FROM events;")
        #expect(res.columns == ["id", "created_at"])
        #expect(res.rows.count == 1)
        #expect(try await reader.isHandleReadOnly(forReadOnlyQuery: true) == true)

        // Reject REPLACE INTO
        await #expect(throws: DatabaseError.self) {
            _ = try await reader.executeReadOnlyQuery("REPLACE INTO events VALUES (1, 'mutated');")
        }

        // Reject writable PRAGMA
        await #expect(throws: DatabaseError.self) {
            _ = try await reader.executeReadOnlyQuery("PRAGMA user_version = 42;")
        }

        // Reject stacked multi-statement query
        await #expect(throws: DatabaseError.self) {
            _ = try await reader.executeReadOnlyQuery("SELECT id FROM events; DELETE FROM events;")
        }
        await reader.close()
    }
}
