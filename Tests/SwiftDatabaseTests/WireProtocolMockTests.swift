import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftDatabase

@Suite("Wire Protocol Driver Mock Server Tests")
struct WireProtocolMockTests {

    /// Helper to start a local TCP mock server on an ephemeral port.
    private static func createMockServer(handler: @escaping (Int32) -> Void) -> (port: Int, stop: () -> Void) {
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
        listen(serverFd, 1)

        var assignedAddr = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        withUnsafeMutablePointer(to: &assignedAddr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                _ = getsockname(serverFd, $0, &len)
            }
        }
        let port = Int(UInt16(bigEndian: assignedAddr.sin_port))

        let isRunning = ManagedAtomicBool(true)

        DispatchQueue.global().async {
            var clientAddr = sockaddr_in()
            var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
            let clientFd = withUnsafeMutablePointer(to: &clientAddr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    accept(serverFd, $0, &clientLen)
                }
            }
            if clientFd >= 0 {
                handler(clientFd)
                close(clientFd)
            }
        }

        let stop = {
            isRunning.value = false
            close(serverFd)
        }

        return (port, stop)
    }

    private final class ManagedAtomicBool: @unchecked Sendable {
        var value: Bool
        init(_ val: Bool) { self.value = val }
    }

    @Test("PostgreSQL trust auth and query response wire decoding")
    func testPostgresTrustAuthAndQuery() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)
            _ = recv(clientFd, &buf, buf.count, 0)

            var authOk = Data([0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00])
            authOk.append(contentsOf: [0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = authOk.withUnsafeBytes { send(clientFd, $0.baseAddress, authOk.count, 0) }

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
            var md5Req = Data([0x52, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x05, 0x01, 0x02, 0x03, 0x04])
            _ = md5Req.withUnsafeBytes { send(clientFd, $0.baseAddress, md5Req.count, 0) }

            // Read password response 'p'
            _ = recv(clientFd, &buf, buf.count, 0)

            // Send AuthenticationOk + ReadyForQuery
            var authOk = Data([0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00])
            authOk.append(contentsOf: [0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = authOk.withUnsafeBytes { send(clientFd, $0.baseAddress, authOk.count, 0) }

            // Read Query
            _ = recv(clientFd, &buf, buf.count, 0)

            // Send ReadyForQuery
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
            var ddlOkPacket = Data([7, 0, 0, 1, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00])
            _ = ddlOkPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, ddlOkPacket.count, 0) }
        }
        defer { stop() }

        let conn = MySQLConnection(connectionURL: "mysql://root:secret@127.0.0.1:\(port)/testdb")
        let res = try await conn.executeQuery("CREATE TABLE t (x INT);")
        #expect(res.rows.isEmpty)
    }
}
