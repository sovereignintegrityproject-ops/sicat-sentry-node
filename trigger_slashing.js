const anchor = require("@coral-xyz/anchor");
const { PublicKey, Keypair } = require("@solana/web3.js");
const fs = require("fs");

async function main() {
    // 1. Setup Connection and Provider
    // This automatically reads your ~/.config/solana/id.json
    const provider = anchor.AnchorProvider.env();
    anchor.setProvider(provider);

    // 2. Load the IDL (The Contract's "Map")
    const idl = JSON.parse(
        fs.readFileSync("./sicat_protocol/target/idl/sicat_protocol.json", "utf8")
    );

    // 3. Connect to the Program
    const programId = new PublicKey("F5K84FMkCoSMpNP967fwAh7ZijrdNtMKv5dwzjaxcuPY"); // Replace with the ID from your 'anchor deploy'
    const program = new anchor.Program(idl, provider);

    console.log("🚀 Sentry Station Online. Program:", programId.toString());

    // 4. Define the Vaults (The 60/35/5 Destinations)
    // For a TEST, we'll use your own wallet address for all of them.
    // In production, these would be separate Treasury/Auditor wallets.
    const myWallet = provider.wallet.publicKey;

    // 5. Trigger the Slashing Event
    const isBiased = true;
    const totalSlashAmount = new anchor.BN(1000000); // 1,000,000 units (e.g., 0.001 SOL or SPL tokens)

    console.log("⚖️ Audit Finding: BIAS DETECTED. Initiating 60/35/5 Split...");

    try {
        const tx = await program.methods
            .settleAudit(isBiased, totalSlashAmount)
            .accounts({
                devStakeVault: myWallet,         // The "Slashed" account
                globalSouthVault: myWallet,      // 60% Destination
                auditorRewardAccount: myWallet,  // 35% Destination
                nodeOpsVault: myWallet,          // 5% Destination
                tokenProgram: anchor.utils.token.TOKEN_PROGRAM_ID,
                systemProgram: anchor.web3.SystemProgram.programId,
            })
            .rpc();

        console.log("✅ Slashing Successfully Logged on Blockchain!");
        console.log("🔗 Transaction Signature:", tx);
    } catch (err) {
        console.error("❌ Error triggering slashing:", err);
    }
}

main();
