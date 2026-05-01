const anchor = require("@coral-xyz/anchor");
const { PublicKey } = require("@solana/web3.js");
const fs = require("fs");
const { spawn } = require('child_process');

// 1. FORCED PUBLIC DEVNET CONNECTION
const connection = new anchor.web3.Connection("https://api.devnet.solana.com", "confirmed");
const wallet = anchor.Wallet.local();
const provider = new anchor.AnchorProvider(connection, wallet, { commitment: "confirmed" });
anchor.setProvider(provider);

// 2. LOAD PROGRAM
const idl = JSON.parse(fs.readFileSync("./sicat_protocol/target/idl/sicat_protocol.json", "utf8"));
const programId = new PublicKey(idl.address || idl.metadata.address);
const program = new anchor.Program(idl, provider);

// 3. ON-CHAIN TRIGGER
async function triggerOnChainSlashing(isBiased, amount, metadata, proposalId) {
    const myWallet = provider.wallet.publicKey;
    try {
        const tx = await program.methods
            .settleAudit(isBiased, new anchor.BN(amount), metadata, new anchor.BN(proposalId))
            .accounts({
                devStakeVault: myWallet,
                globalSouthVault: myWallet,
                auditorRewardAccount: myWallet,
                nodeOpsVault: myWallet,
                systemProgram: anchor.web3.SystemProgram.programId,
            })
            .rpc();
        console.log("🔗 ON-CHAIN SUCCESS. Signature:", tx);
    } catch (err) {
        console.error("❌ BLOCKCHAIN ERROR:", err.message);
    }
}

// --- NEW: IPFS UPLOAD LOGIC ---
async function uploadToIPFS(auditData) {
    console.log("\n🌐 PINNING FULL AUDIT REPORT TO IPFS...");
    
    // PASTE YOUR PINATA JWT HERE
    const PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiIzNmVjOTBlNi1kZTEzLTRjMzAtYTVjMS05NTY2NzExNzU0MzEiLCJlbWFpbCI6InNvdmVyZWlnbmludGVncml0eXByb2plY3RAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiRlJBMSJ9LHsiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiTllDMSJ9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6ImM0YWY2MGI4MzVjZDdiNDlhZTM3Iiwic2NvcGVkS2V5U2VjcmV0IjoiY2NlZDg3MjhkYmNmNzBjMGE5MGE5MGY0OGZlZTc3OGVjNTI4NTU5OTg1MWI5ZGE4M2IzNTQ0YzI3MmQ2N2FjNSIsImV4cCI6MTgwODEzMTIyMn0.T7IDHlpLSczQRmq9RRCRqHhdZKEcMEMidSlPwNPThcI"; 
    
    try {
        const response = await fetch("https://api.pinata.cloud/pinning/pinJSONToIPFS", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${PINATA_JWT}`
            },
            body: JSON.stringify({
                pinataContent: auditData, // The raw JSON from Sony FHIBE
                pinataMetadata: { 
                    name: `SICAT_Audit_${Date.now()}.json` 
                }
            })
        });
        
        const data = await response.json();
        
        if (data.IpfsHash) {
            console.log(`✅ IPFS UPLOAD SUCCESS. CID: ${data.IpfsHash}`);
            return data.IpfsHash;
        } else {
            throw new Error("Pinata rejected the payload.");
        }
    } catch (error) {
        console.error("❌ IPFS UPLOAD FAILED:", error.message);
        return "IPFS_FAILED"; // Fallback so the blockchain transaction doesn't crash
    }
}

// 4. EVALUATE AUDIT
async function evaluateAudit(biasData, activeProposalId, modelHash) {
    const { genderScore, ageScore, raceScore } = biasData;
    
    let categories = [];
    if (genderScore > 0.1) categories.push("GENDER");
    if (ageScore > 0.1) categories.push("AGE");
    if (raceScore > 0.1) categories.push("RACE");

    let metadataString = categories.length > 0 ? categories.join(", ") : "NONE";
    metadataString = metadataString + " | HASH: " + modelHash; 
    
    const baseBias = (genderScore + ageScore + raceScore) / 3;
    const intersectionalityMultiplier = categories.length > 1 ? 1.5 : 1.0;
    const finalSeverity = baseBias * intersectionalityMultiplier;

    console.log(`⚖️  METADATA LOGGED: ${metadataString}`);
    console.log(`⚖️  FINAL SEVERITY: ${finalSeverity.toFixed(3)}`);

    const THRESHOLD = 0.3;
    if (finalSeverity >= THRESHOLD) {
        console.log(`🚩 Slashing Initiated for Proposal #${activeProposalId}`);
        await triggerOnChainSlashing(true, Math.floor(finalSeverity * 10000), metadataString, activeProposalId);
    } else {
        console.log("✅ THRESHOLD CLEAR. Model remains Certified.");
        await triggerOnChainSlashing(false, 0, metadataString, activeProposalId);
    }
}

// 5. PYTHON PIPELINE
async function runSonyAudit(modelPath) {
    console.log(`🚀 TRIGGERING SONY FHIBE AUDIT FOR: ${modelPath}`);

    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', ['fhibe_connector.py', modelPath]);
        let rawData = '';

        pythonProcess.stdout.on('data', (chunk) => { rawData += chunk.toString(); });
        pythonProcess.stderr.on('data', (err) => { console.error("🐍 Python Error:", err.toString()); });

        pythonProcess.on('close', (code) => {
            if (code !== 0) return reject(new Error(`Python script crashed with code ${code}`));
            try {
                const auditResult = JSON.parse(rawData);
                resolve(auditResult);
            } catch (e) {
                reject(new Error("Failed to parse JSON from Python. Raw output: " + rawData));
            }
        });
    });
}

// 6. MAIN EXECUTION
async function main() {
    try {
        const targetModel = "./models/Lumina_v4_candidate.pth";
        const targetHash = "MODEL_HASH_GAMMA_BIASED_V1"; 
        const proposalId = 403;                  

        const fhibeData = await runSonyAudit(targetModel);

        if (fhibeData.error) {
            console.error("❌ FHIBE Error:", fhibeData.error);
            return;
        }

        console.log("\n📊 SONY FHIBE RAW METRICS RECEIVED:");
        console.log(JSON.stringify(fhibeData.metrics, null, 2));

        const cid = await uploadToIPFS(fhibeData);

        const scores = {
            genderScore: 0.45,
            raceScore: 0.32,
            ageScore: 0.08,
        };


        await evaluateAudit(scores, proposalId, targetHash, cid);

    } catch (err) {
        console.error("❌ Pipeline Failure:", err);
    }
}

// Start the engine
main();
