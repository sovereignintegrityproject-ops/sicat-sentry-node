import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import * as snarkjs from "npm:snarkjs";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const { session_id, nullifier_hash, identity_commitment, zk_proof, justification_cid } = await req.json();

    // 1. FETCH THE LOCK
    const VKEY_URL = "https://rvyuagsiibwnwqsyqusf.supabase.co/storage/v1/object/public/zk-artifacts/verification_key.json";
    const vKeyRes = await fetch(VKEY_URL);
    const vKey = await vKeyRes.json();

    // 2. CHECK THE MATH
    const publicSignals = [identity_commitment, nullifier_hash, session_id];
    console.log("Verifying Cryptography...");
    
    const isValid = await snarkjs.groth16.verify(vKey, publicSignals, zk_proof);

    if (!isValid) {
      throw new Error("Zero-Knowledge Proof invalid.");
    }

    console.log("Math Verified. Opening Vault...");

    // 3. INSERT INTO DATABASE
    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const { error: insertError } = await supabaseAdmin
      .from('veto_signatures')
      .insert([{
        session_id,
        nullifier_hash,
        identity_commitment,
        zk_proof,
        justification_cid
      }]);

    if (insertError) {
      if (insertError.code === "23505") throw new Error("A vote with this Nullifier Hash has already been cast.");
      throw insertError;
    }

    // 4. PING THE UBUNTU BACKEND
    console.log("Vault Opened. Pinging Backend Engine via Wormhole...");
    const FHIBE_BACKEND_URL = "https://picking-literature-extends-restrictions.trycloudflare.com/execute-fhibe";
    
    try {
      const backendRes = await fetch(FHIBE_BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id, nullifier_hash }),
      });
      console.log("Wormhole response received.");
    } catch (e) {
      console.error("Wormhole signal lost, but vote recorded locally:", e);
    }

    // 5. FINAL SUCCESS RESPONSE TO PHONE
    return new Response(JSON.stringify({ success: true, message: "Veto successfully cast and engine signaled." }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error) {
    console.error("🔴 Bouncer Rejected:", error.message);
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    });
  }
});
