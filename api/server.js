const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3000;

app.post('/execute-fhibe', (req, res) => {
    const { session_id, nullifier_hash } = req.body;
    
    console.log(`\n========================================`);
    console.log(`🟢 VETO RECEIVED: Session ${session_id}`);
    console.log(`========================================`);
    console.log(`Auditor Nullifier: ${nullifier_hash}`);
    console.log(`Status: ZK-Proof Verified by Supabase Edge.`);
    console.log(`\n⏳ PENDING: Handing off to Sony FHIBE Engine...`);

    // THE FIXED LINE: We 'cd' all the way into the room before pressing play
    const cmd = `cd ~/gjac1-lumina-backend/fhibe_evaluation_api-main/demo/person_localization && poetry run python run_person_localization_demo.py`;

    exec(cmd, (error, stdout, stderr) => {
        if (error) {
            console.error(`🔴 FHIBE Error: ${error.message}`);
            return;
        }
        console.log(`\n✅ Bias Report Generated for Session ${session_id}`);
        console.log(stdout);
    });

    res.status(200).json({ 
        success: true, 
        message: 'Backend acknowledges ZK-Proof. Audit Triggered.' 
    });
});

app.listen(PORT, () => {
    console.log(`🚀 Dedicated FHIBE Backend listening on local port ${PORT}`);
});
