async function main() {
    // 1. Force 'confirmed' commitment
    const connection = new anchor.web3.Connection(
        process.env.ANCHOR_PROVIDER_URL || "http://localhost:8899", 
        "confirmed"
    );
    const wallet = anchor.Wallet.local();
    const provider = new anchor.AnchorProvider(connection, wallet, { commitment: "confirmed" });
    anchor.setProvider(provider);
    
    // ... (rest of the code)
}
const anchor = require("@coral-xyz/anchor");
const { PublicKey } = require("@solana/web3.js");
const fs = require("fs");

async function main() {
    // 1. Setup Connection
    const provider = anchor.AnchorProvider.env();
    provider.connection.commitment = "confirmed"; // Ensures we see the data faster
    anchor.setProvider(provider);

    // 2. Program Setup
    const idl = JSON.parse(fs.readFileSync("./sicat_protocol/target/idl/sicat_protocol.json", "utf8"));
    const programId = new PublicKey("F5K84FMkCoSMpNP967fwAh7ZijrdNtMKv5dwzjaxcuPY");
    const program = new anchor.Program(idl, provider);

    console.log("📊 SENTRY LEDGER: Fetching Historical Reparations...");

    // 3. The "Proposal Tracker" - An object to store our totals
    const fundSummary = {};

    // 4. Fetch the events (Note: In a production environment with thousands of events, 
    // we would use a dedicated indexer like Helius or Shyft. For now, we pull from logs).
    const eventParser = new anchor.EventParser(programId, program.coder);
    
    // Get the last 100 signatures (adjust as needed)
    const signatures = await provider.connection.getSignaturesForAddress(programId, { limit: 100 });

    for (let sigInfo of signatures) {
        const tx = await provider.connection.getTransaction(sigInfo.signature, {
            commitment: "confirmed",
            maxSupportedTransactionVersion: 0
        });

        if (tx && tx.meta && tx.meta.logMessages) {
            const events = eventParser.parseLogs(tx.meta.logMessages);
            
            for (let event of events) {
                if (event.name === "AuditSettled") {
                    const { amount, proposalId, categories } = event.data;
                    const pId = proposalId.toString();

                    if (!fundSummary[pId]) {
                        fundSummary[pId] = { total: 0, categories: new Set() };
                    }

                    // Aggregate the data
                    fundSummary[pId].total += amount.toNumber();
                    categories.split(", ").forEach(cat => fundSummary[pId].categories.add(cat));
                }
            }
        }
    }

    // 5. Output the Results in a clean table
    console.log("\n--- GLOBAL SOUTH BUILDING FUND REPORT ---");
    console.table(Object.keys(fundSummary).map(pId => ({
        "Proposal ID": `#${pId}`,
        "Total Raised (Units)": fundSummary[pId].total,
        "Primary Bias Drivers": Array.from(fundSummary[pId].categories).join(", ")
    })));
    console.log("------------------------------------------");
}

main().catch(err => console.error(err));
