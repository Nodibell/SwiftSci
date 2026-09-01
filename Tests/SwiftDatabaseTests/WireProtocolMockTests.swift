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
            // 1. Read StartupMessage
            _ = recv(clientFd, &buf, buf.count, 0)

            // 2. Send AuthenticationOk ('R' + len(8) + type(0)) + ReadyForQuery ('Z' + len(5) + 'I')
            var authOk = Data([0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00])
            authOk.append(contentsOf: [0x5A, 0x00, 0x00, 0x00, 0x05, 0x49])
            _ = authOk.withUnsafeBytes { send(clientFd, $0.baseAddress, authOk.count, 0) }

            // 3. Read Query ('Q')
            _ = recv(clientFd, &buf, buf.count, 0)

            // 4. Send RowDescription ('T') + DataRow ('D') + ReadyForQuery ('Z')
            var resp = Data()

            // RowDescription: col 'id' and col 'name'
            var tBody = Data([0x00, 0x02]) // 2 columns
            // col 1: "id\0" + 18 bytes field info
            tBody.append(contentsOf: "id\0".utf8)
            tBody.append(contentsOf: [UInt8](repeating: 0, count: 18))
            // col 2: "name\0" + 18 bytes field info
            tBody.append(contentsOf: "name\0".utf8)
            tBody.append(contentsOf: [UInt8](repeating: 0, count: 18))

            resp.append(0x54) // 'T'
            withUnsafeBytes(of: Int32(tBody.count + 4).bigEndian) { resp.append(contentsOf: $0) }
            resp.append(tBody)

            // DataRow: (1, "Alice")
            var dBody = Data([0x00, 0x02]) // 2 values
            // val 1: "1"
            let val1 = "1"
            withUnsafeBytes(of: Int32(val1.utf8.count).bigEndian) { dBody.append(contentsOf: $0) }
            dBody.append(contentsOf: val1.utf8)
            // val 2: "Alice"
            let val2 = "Alice"
            withUnsafeBytes(of: Int32(val2.utf8.count).bigEndian) { dBody.append(contentsOf: $0) }
            dBody.append(contentsOf: val2.utf8)

            resp.append(0x44) // 'D'
            withUnsafeBytes(of: Int32(dBody.count + 4).bigEndian) { resp.append(contentsOf: $0) }
            resp.append(dBody)

            // ReadyForQuery
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

    @Test("MySQL wire handshake and query response wire decoding")
    func testMySQLHandshakeAndQuery() async throws {
        let (port, stop) = Self.createMockServer { clientFd in
            var buf = [UInt8](repeating: 0, count: 4096)

            // 1. Send Handshake packet (v10)
            var handshakePayload = Data([10]) // proto v10
            handshakePayload.append(contentsOf: "8.0.32\0".utf8)
            handshakePayload.append(contentsOf: [1, 0, 0, 0]) // thread id
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x41, count: 8)) // salt 1
            handshakePayload.append(0x00) // filler
            handshakePayload.append(contentsOf: [0xFF, 0xF7]) // cap 1
            handshakePayload.append(33) // charset utf8
            handshakePayload.append(contentsOf: [0x02, 0x00]) // status
            handshakePayload.append(contentsOf: [0xFF, 0x81]) // cap 2
            handshakePayload.append(21) // auth plugin data len
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x00, count: 10)) // reserved
            handshakePayload.append(contentsOf: [UInt8](repeating: 0x42, count: 12)) // salt 2
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
            // Packet 1: Column Count = 1
            var colCountPacket = Data([1, 0, 0, 1, 1])
            _ = colCountPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, colCountPacket.count, 0) }

            // Packet 2: ColumnDefinition for "score"
            // catalog(def), schema(db), table(tbl), org_table(tbl), name(score)
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
            colDefPayload.append(contentsOf: [UInt8](repeating: 0, count: 12)) // type flags

            let cdLen = colDefPayload.count
            var colDefPacket = Data([UInt8(cdLen & 0xFF), UInt8((cdLen >> 8) & 0xFF), UInt8((cdLen >> 16) & 0xFF), 2])
            colDefPacket.append(colDefPayload)
            _ = colDefPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, colDefPacket.count, 0) }

            // Packet 3: EOF packet
            var eofPacket = Data([5, 0, 0, 3, 0xFE, 0x00, 0x00, 0x02, 0x00])
            _ = eofPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, eofPacket.count, 0) }

            // Packet 4: Row 1 ("99.5")
            var rowPayload = Data()
            rowPayload.append(UInt8("99.5".utf8.count))
            rowPayload.append(contentsOf: "99.5".utf8)
            let rLen = rowPayload.count
            var rowPacket = Data([UInt8(rLen & 0xFF), UInt8((rLen >> 8) & 0xFF), UInt8((rLen >> 16) & 0xFF), 4])
            rowPacket.append(rowPayload)
            _ = rowPacket.withUnsafeBytes { send(clientFd, $0.baseAddress, rowPacket.count, 0) }

            // Packet 5: EOF packet
            var eofPacket2 = Data([5, 0, 0, 5, 0xFE, 0x00, 0x00, 0x02, 0x00])
            _ = eofPacket2.withUnsafeBytes { send(clientFd, $0.baseAddress, eofPacket2.count, 0) }
        }
        defer { stop() }

        let conn = MySQLConnection(connectionURL: "mysql://root:secret@127.0.0.1:\(port)/testdb")
        let result = try await conn.executeQuery("SELECT score FROM orders;")
        #expect(result.columns == ["score"])
        #expect(result.rows.count == 1)
        #expect(result.rows[0][0].description == "99.5")
    }
}
