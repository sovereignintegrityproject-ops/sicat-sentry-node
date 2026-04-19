const { exec } = require('child_process');

app.post('/execute-fhibe', (req, res) => {
    const { session_id, nullifier_hash } = req.body;
    
    console.log(`🟢 VETO RECEIVED: Session ${session_id}`);

    // This command tells Poetry to run the Sony Evaluation script
    // We point it at the 'person_localization' demo as a starting test
    const cmd = `cd ~/gjac1-lumina-backend/fhibe_evaluation_api-main && poetry run python demo/person_localization/run_person_localization.py`;

    exec(cmd, (error, stdout, stderr) => {
        if (error) {
            console.error(`🔴 FHIBE Error: ${error.message}`);
            return;
        }
        console.log(`✅ Bias Report Generated for Session ${session_id}`);
        console.log(stdout);
    });

    res.status(200).json({ success: true, message: 'Audit Triggered.' });
});

app.listen(PORT, () => {
    console.log(`🚀 Dedicated FHIBE Backend listening on local port ${PORT}`);
});
