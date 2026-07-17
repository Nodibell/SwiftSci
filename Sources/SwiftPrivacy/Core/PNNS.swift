import Foundation

/// A client for Private Nearest Neighbor Search.
/// Holds the secret key and encrypts query vectors.
public struct PNNSClient: Sendable {
    public let key: LWESecretKey
    
    public init(key: LWESecretKey) {
        self.key = key
    }
    
    /// Encrypts a query vector coordinate-wise.
    public func encryptQuery(_ query: [Int]) -> [LWECiphertext] {
        return query.map { LWE.encrypt(message: $0, key: key) }
    }
    
    /// Decrypts a list of secure distances or dot products.
    public func decryptResults(_ ciphertexts: [LWECiphertext]) -> [Int] {
        return ciphertexts.map { key.decrypt($0) }
    }
    
    /// Finds the index of the nearest neighbor (having the minimum decrypted Euclidean distance).
    public func findNearestNeighborIndex(decryptedDistances: [Int]) -> Int? {
        guard !decryptedDistances.isEmpty else { return nil }
        
        var minDistance = Int.max
        var minIndex = 0
        
        for (idx, dist) in decryptedDistances.enumerated() {
            if dist < minDistance {
                minDistance = dist
                minIndex = idx
            }
        }
        
        return minIndex
    }
    
    /// Finds the index of the most similar neighbor (having the maximum decrypted dot product).
    public func findMostSimilarIndex(decryptedDotProducts: [Int]) -> Int? {
        guard !decryptedDotProducts.isEmpty else { return nil }
        
        var maxDot = Int.min
        var maxIndex = 0
        
        for (idx, dot) in decryptedDotProducts.enumerated() {
            if dot > maxDot {
                maxDot = dot
                maxIndex = idx
            }
        }
        
        return maxIndex
    }
}

/// A server for Private Nearest Neighbor Search.
/// Holds the database of vector records and computes homomorphic distances.
public struct PNNSServer: Sendable {
    public let database: [[Int]]
    
    public init(database: [[Int]]) {
        self.database = database
    }
    
    /// Computes the secure dot product between the encrypted query vector and all database vectors.
    /// - Parameter encryptedQuery: Coordination-wise LWE-encrypted query.
    /// - Returns: A list of encrypted dot products.
    public func computeSecureDotProducts(encryptedQuery: [LWECiphertext]) -> [LWECiphertext] {
        var results = [LWECiphertext]()
        results.reserveCapacity(database.count)
        
        for row in database {
            guard !row.isEmpty && row.count == encryptedQuery.count else { continue }
            
            // secureDot = sum_{j} x_j * Enc(q_j)
            var secureDot = LWE.multiply(encryptedQuery[0], by: row[0])
            for j in 1..<row.count {
                let prod = LWE.multiply(encryptedQuery[j], by: row[j])
                secureDot = LWE.add(secureDot, prod)
            }
            
            results.append(secureDot)
        }
        
        return results
    }
    
    /// Computes the secure squared Euclidean distance between the encrypted query and database vectors.
    /// To do this homomorphically:
    /// dist^2 = sum_j (q_j - x_j)^2 = sum_j q_j^2 - 2 * sum_j x_j * q_j + sum_j x_j^2
    /// Client sends BOTH Enc(q) and Enc(q^2) coordinates.
    /// - Parameters:
    ///   - encryptedQuery: Encrypted q vector coordinate-wise.
    ///   - encryptedQuerySquared: Encrypted q^2 vector coordinate-wise.
    ///   - clientKey: The client's public metadata (e.g. key structure containing scaling factor delta).
    /// - Returns: A list of encrypted squared Euclidean distances.
    public func computeSecureEuclideanDistances(
        encryptedQuery: [LWECiphertext],
        encryptedQuerySquared: [LWECiphertext],
        clientKey: LWESecretKey
    ) -> [LWECiphertext] {
        var results = [LWECiphertext]()
        results.reserveCapacity(database.count)
        
        for row in database {
            guard !row.isEmpty && row.count == encryptedQuery.count else { continue }
            
            // 1. Homomorphic sum of q_j^2
            var secureDistance = encryptedQuerySquared[0]
            for j in 1..<row.count {
                secureDistance = LWE.add(secureDistance, encryptedQuerySquared[j])
            }
            
            // 2. Homomorphic subtraction of 2 * x_j * q_j
            for j in 0..<row.count {
                let doubleX = 2 * row[j]
                let prod = LWE.multiply(encryptedQuery[j], by: doubleX)
                // In LWE: Enc(A) - Enc(B) = Enc(A) + Enc(-B)
                let negProd = LWE.multiply(prod, by: -1)
                secureDistance = LWE.add(secureDistance, negProd)
            }
            
            // 3. Homomorphic addition of plain sum of x_j^2
            let sumXSq = row.map { $0 * $0 }.reduce(0, +)
            secureDistance = LWE.addPlain(secureDistance, sumXSq, key: clientKey)
            
            results.append(secureDistance)
        }
        
        return results
    }
}
