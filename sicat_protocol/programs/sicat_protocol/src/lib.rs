use anchor_lang::prelude::*;

declare_id!("F5K84FMkCoSMpNP967fwAh7ZijrdNtMKv5dwzjaxcuPY");

#[program]
pub mod sicat_protocol {
    use super::*;

    pub fn settle_audit(
        _ctx: Context<SettleAudit>, 
        is_biased: bool, 
        total_slash_amount: u64,
        metadata: String, // <--- New Field
        proposal_id: u64, // <--- Added Proposal ID
    ) -> Result<()> {
        if is_biased {
            let fund_share = (total_slash_amount * 6000) / 10000;
            let auditor_share = (total_slash_amount * 3500) / 10000;
            let ops_share = total_slash_amount - fund_share - auditor_share;

            msg!("Bias Detected! Categories: {}", metadata);
            msg!("Distribution: Fund: {}, Auditor: {}, Ops: {}", fund_share, auditor_share, ops_share);

            // Emit an event for the Dashboard to pick up
            emit!(AuditSettled {
                amount: total_slash_amount,
                categories: metadata,
                proposal_id,
                timestamp: Clock::get()?.unix_timestamp,
            });
        }
        Ok(())
    }
}

#[derive(Accounts)]
pub struct SettleAudit<'info> {
    /// CHECK: Testing purposes - skipping ownership check
    #[account(mut)]
    pub dev_stake_vault: UncheckedAccount<'info>,
    
    /// CHECK: Testing purposes - skipping ownership check
    #[account(mut)]
    pub global_south_vault: UncheckedAccount<'info>,
    
    /// CHECK: Testing purposes - skipping ownership check
    #[account(mut)]
    pub auditor_reward_account: UncheckedAccount<'info>,
    
    /// CHECK: Testing purposes - skipping ownership check
    #[account(mut)]
    pub node_ops_vault: UncheckedAccount<'info>,
    
    pub system_program: Program<'info, System>,
}

#[event]
pub struct AuditSettled {
    pub amount: u64,
    pub categories: String,
    pub proposal_id: u64,
    pub timestamp: i64,
}

// Add this to your events at the bottom
#[event]
pub struct AuditPassed {
    pub model_hash: String,
    pub timestamp: i64,
}
