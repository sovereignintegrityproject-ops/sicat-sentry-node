const anchor = require("@coral-xyz/anchor");
const { PublicKey } = require("@solana/web3.js");
const fs = require("fs");
const qrcode = require("qrcode-terminal");

// Added a second parameter for the exact signature
async function checkIntegrity(targetHash, knownSignature) {
    // 1. Pointed strictly to Devnet
    const connection = new anchor.web3.Connection("https://api.devnet.solana.com", "confirmed");
    const wallet = anchor.Wallet.local();
    const provider = new anchor.AnchorProvider(connection, wallet, { commitment: "confirmed" });
    anchor.setProvider(provider);

    const idl = JSON.parse(fs.readFileSync("./sicat_protocol/target/idl/sicat_protocol.json", "utf8"));
    const programId = new PublicKey(idl.address || idl.metadata.address);
    const program = new anchor.Program(idl, provider);

    console.log(`🔎 DIRECT SCAN FOR HASH: ${targetHash}`);
    console.log(`🎯 SNIPER MODE: Fetching signature...`);

    const eventParser = new anchor.EventParser(programId, program.coder);

    let status = "⚠️ UNKNOWN";
    let proofUrl = "";

    // 2. Fetch the specific transaction directly!
    const tx = await connection.getTransaction(knownSignature, { 
        commitment: "confirmed", 
        maxSupportedTransactionVersion: 0 
    });

    if (!tx) {
        console.log("❌ CRITICAL: Transaction not found. Validator may have restarted.");
        return;
    }

    const events = eventParser.parseLogs(tx.meta.logMessages);
    for (let event of events) {
        if (event.name === "auditSettled" && event.data.categories.includes(targetHash)) {
            status = "❌ SLASHED (Bias Detected)";
            proofUrl = `https://explorer.solana.com/tx/${knownSignature}?cluster=custom&customUrl=http%3A%2F%2Flocalhost%3A8899`;
        } else if (event.name === "auditPassed" && (event.data.modelHash === targetHash || event.data.model_hash === targetHash)) {
            status = "✅ CERTIFIED (Integrity Verified)";
            proofUrl = `https://explorer.solana.com/tx/${knownSignature}?cluster=devnet`;
        }
    }

    console.log(`==========================================`);
    console.log(`RESULT: ${status}`);
    if (proofUrl) {
        console.log("🔗 ON-CHAIN PROOF:");
        qrcode.generate(proofUrl, { small: true });
        console.log(proofUrl);
    }
    console.log(`==========================================\n`);
}

// 3. Paste the EXACT signature your Engine generated earlier!
checkIntegrity(
    "MODEL_HASH_GAMMA_9", 
    "SPqxKUhGfL2ZXVMNYUscmNcYFCsgSUQxkDujUhNVf9UVQSLNeT94dRm7dXz2MeWzve3Rc6S7MSiV3g1RSxfZfti"
);
