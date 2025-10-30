# export_ngos_to_pdf.py
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

OUTPUT_DIR = "./ngo_profiles"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=6, spaceAfter=4, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Label", parent=styles["BodyText"], spaceBefore=6, spaceAfter=2, leading=13))
styles.add(ParagraphStyle(name="BodyTight", parent=styles["BodyText"], spaceBefore=0, spaceAfter=4, leading=13))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12, spaceBefore=0, spaceAfter=3))

def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-", ".")).rstrip().replace(" ", "_")

def add_label_value(story, label: str, value: str):
    story.append(Paragraph(f"<b>{label}:</b> {value}", styles["BodyTight"]))

def add_list(story, label: str, items: list):
    story.append(Paragraph(f"<b>{label}:</b>", styles["Label"]))
    bullet_items = [ListItem(Paragraph(it, styles["BodyTight"])) for it in items]
    story.append(ListFlowable(bullet_items, bulletType="bullet", start="•", leftIndent=12))
    story.append(Spacer(1, 4))

def add_table_kpis(story, title: str, rows: list):
    # rows is a list of tuples: (KPI, Baseline, Latest, Target)
    story.append(Paragraph(f"<b>{title}:</b>", styles["Label"]))
    data = [["KPI", "Baseline", "Latest", "Target"]] + [list(r) for r in rows]
    tbl = Table(data, colWidths=[70*mm, 30*mm, 30*mm, 30*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))

def add_section_header(story, text: str):
    story.append(Paragraph(text, styles["H2"]))
    story.append(Spacer(1, 2))

def write_pdf(profile: dict):
    title = profile["title"]
    filename = sanitize_filename(f"{title}.pdf")
    path = os.path.join(OUTPUT_DIR, filename)
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm
    )
    story = []

    # Title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 6))

    # Summary
    add_label_value(story, "Mission", profile["mission"])
    add_list(story, "Focus areas", profile["focus_areas"])
    add_list(story, "Programs", profile["programs"])
    add_label_value(story, "Geographic focus", profile["geographic_focus"])
    add_label_value(story, "Beneficiaries", profile["beneficiaries"])
    add_list(story, "Annual impact highlights", profile["impact"])

    # Registration & Compliance
    add_section_header(story, "Registration & Compliance")
    add_list(story, "Registrations", profile["registration"]["registrations"])
    add_list(story, "Policies", profile["registration"]["policies"])
    add_label_value(story, "Governance", profile["governance"])

    # SDG Alignment
    add_section_header(story, "SDG Alignment")
    add_list(story, "Relevant SDGs", profile["sdgs"])

    # Monitoring & Evaluation
    add_section_header(story, "Monitoring & Evaluation")
    add_list(story, "M&E Approach", profile["monitoring"])

    # Risks & Mitigations
    add_section_header(story, "Key Risks & Mitigations")
    add_list(story, "Risks", [f"{r['risk']} — {r['mitigation']}" for r in profile["risks"]])

    # Partnerships
    add_section_header(story, "Partnerships")
    add_list(story, "Key partners", profile["partnerships"])

    # KPIs Table
    add_table_kpis(story, "Operational KPIs", profile["kpis"])

    # Case Study
    add_section_header(story, "Case Study")
    story.append(Paragraph(profile["case_study"], styles["BodyTight"]))
    story.append(Spacer(1, 6))

    # FAQs
    add_section_header(story, "FAQs")
    for qa in profile["faq"]:
        story.append(Paragraph(f"<b>Q:</b> {qa['q']}", styles["BodyTight"]))
        story.append(Paragraph(f"<b>A:</b> {qa['a']}", styles["Small"]))
        story.append(Spacer(1, 2))

    # Volunteer & Donate
    add_section_header(story, "Volunteer & Donate")
    add_list(story, "Volunteer opportunities", profile["volunteer"])
    add_label_value(story, "Donate", profile["donate"])

    # Financials & Contact
    add_section_header(story, "Financials & Contact")
    add_label_value(story, "Financial snapshot", profile["financials"])
    add_label_value(story, "Contact", profile["contact"])

    # Footnote
    story.append(Spacer(1, 8))
    story.append(Paragraph("Note: Sample profile for testing and evaluation.", styles["Small"]))

    doc.build(story)
    return path

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    profiles = [
        {
            "title": "Paws & Claws Rescue Alliance",
            "mission": "Improve animal welfare by rescuing, rehabilitating, and rehoming abandoned companion animals while advancing community-based sterilization and vaccination.",
            "focus_areas": ["Animal rescue", "Shelter management", "ABC-ARV programs", "Community adoption drives"],
            "programs": ["Mobile rescue and first-aid vans", "City shelter with foster network", "Mass sterilization and ARV", "School awareness sessions"],
            "geographic_focus": "Maharashtra and Goa (Pune, Thane, Panaji)",
            "beneficiaries": "Street dogs and cats, low-income pet guardians, and urban communities",
            "impact": ["2,800 rescues", "4,500 ABC surgeries", "12,000 ARV doses", "1,100 adoptions", "90 school sessions"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Not registered"],
                "policies": ["Safeguarding & Anti-Harassment", "Whistleblower", "Data Privacy", "Shelter Biosecurity SOPs"]
            },
            "governance": "Independent board with quarterly program reviews and audited financials",
            "sdgs": ["SDG3 Good Health & Well-being", "SDG11 Sustainable Cities"],
            "monitoring": [
                "Monthly shelter dashboards (intake, outcomes, live release rate)",
                "Surgery quality audits and post-op checks",
                "Community complaint-to-resolution tracking"
            ],
            "risks": [
                {"risk": "Disease outbreaks in kennels", "mitigation": "Strict intake quarantine and vaccination protocols"},
                {"risk": "Adoption drop-offs", "mitigation": "Foster-to-adopt pipelines and post-adoption support"}
            ],
            "partnerships": ["Municipal Corporations", "Veterinary colleges", "Local feeders networks"],
            "kpis": [
                ("Live Release Rate", "78%", "86%", "90%"),
                ("Median Days to Adoption", "42", "28", "21"),
                ("Post-op Complication Rate", "4.0%", "2.2%", "<2%")
            ],
            "case_study": "After mass ARV and targeted ABC across two wards, reported dog-bite incidents fell by 34% over 12 months while adoption rates doubled.",
            "faq": [
                {"q": "Do you accept injured animals at night?", "a": "Yes, the rescue helpline operates 24×7 with on-call volunteers."},
                {"q": "What does INR 2,000 sponsor?", "a": "A sterilization surgery with pre/post-operative care and vaccination."}
            ],
            "volunteer": ["Fostering animals for 2–6 weeks", "Shelter cleaning and enrichment", "Adoption events support"],
            "donate": "UPI and bank transfer; receipts issued; 80G eligible",
            "financials": "FY 2023–24: INR 4.1 Cr; 82% program, 10% admin, 8% fundraising",
            "contact": "info@pawsclaws.org | Pune HQ"
        },
        {
            "title": "EnableTech Foundation",
            "mission": "Accelerate disability inclusion through assistive technology access, inclusive education support, and workforce enablement.",
            "focus_areas": ["Assistive devices", "Inclusive classrooms", "Digital accessibility", "Job skilling"],
            "programs": ["Subsidized devices and repair camps", "Inclusion toolkits and teacher training", "Accessible labs and screen-reader training", "Employer sensitization and placement"],
            "geographic_focus": "Karnataka, Tamil Nadu, Telangana",
            "beneficiaries": "Persons with visual, hearing, and locomotor impairments, schools, and employers",
            "impact": ["2,300 devices provisioned", "540 teachers trained", "780 youth placed", "120 public websites remediated"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: In progress"],
                "policies": ["Accessibility & Inclusion", "Child & Vulnerable Adult Protection", "Data Privacy"]
            },
            "governance": "Disability-led advisory council; quarterly outcomes published",
            "sdgs": ["SDG4 Quality Education", "SDG8 Decent Work", "SDG10 Reduced Inequalities"],
            "monitoring": [
                "Device impact follow-ups at 3 and 9 months",
                "Placement retention tracked at 6 and 12 months",
                "Digital accessibility remediation QA checks"
            ],
            "risks": [
                {"risk": "Low device utilization", "mitigation": "User training and service helpline"},
                {"risk": "Placement attrition", "mitigation": "Mentoring and workplace accommodations"}
            ],
            "partnerships": ["District Disability Offices", "ITI/Polytechnic institutes", "Employer councils"],
            "kpis": [
                ("Device Usage After 6m", "—", "78%", "85%"),
                ("Placement Retention 12m", "—", "71%", "80%"),
                ("WCAG AA Compliance Rate", "54%", "76%", "90%")
            ],
            "case_study": "Screen-reader training plus employer sensitization enabled 40 visually impaired youth to secure QA roles with 85% 12-month retention.",
            "faq": [
                {"q": "Which devices are subsidized?", "a": "Screen readers, hearing aids, mobility aids, and select communication devices."},
                {"q": "Do you support school inclusion?", "a": "Yes, via teacher toolkits, classroom audits, and low-cost accommodations."}
            ],
            "volunteer": ["Accessibility testing for websites", "Career mentorship", "Device camps logistics"],
            "donate": "CSR device kits and scholarship support; 80G eligible",
            "financials": "FY 2023–24: INR 3.6 Cr; 85% program, 8% admin, 7% fundraising",
            "contact": "hello@enabletech.in | Bengaluru"
        },
        {
            "title": "Rapid Relief Network",
            "mission": "Deliver fast, last-mile disaster response and build resilient communities through preparedness and early recovery.",
            "focus_areas": ["Disaster response", "WASH kits", "Shelter repair", "Community preparedness"],
            "programs": ["48-hour rapid response logistics", "Safe water and hygiene kits", "Cash-for-work debris clearing", "Village disaster committees and drills"],
            "geographic_focus": "Odisha, Assam, Uttarakhand",
            "beneficiaries": "Flood, cyclone, and landslide affected households—priority to women and children",
            "impact": ["32 deployments", "18,000 households served", "650 volunteers trained", "220 shelters repaired"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Registered"],
                "policies": ["Humanitarian Safeguarding", "Anti-Fraud & Corruption", "Data Privacy"]
            },
            "governance": "Sphere standards adopted; third-party PDMs post response",
            "sdgs": ["SDG6 Clean Water & Sanitation", "SDG11 Sustainable Cities", "SDG13 Climate Action"],
            "monitoring": [
                "Response time benchmarks (48h target)",
                "Post-distribution monitoring (PDM) surveys",
                "Cash-for-work verification and audits"
            ],
            "risks": [
                {"risk": "Access constraints", "mitigation": "Local partners and pre-positioned stocks"},
                {"risk": "Leakage of relief kits", "mitigation": "Barcode-based inventory and spot checks"}
            ],
            "partnerships": ["State Disaster Management Authorities", "WASH clusters", "Local CBO networks"],
            "kpis": [
                ("Avg Response Time", "72h", "44h", "<=48h"),
                ("Households Reached/Deployment", "—", "560", "600"),
                ("PDM Satisfaction", "—", "91%", "90%+")
            ],
            "case_study": "In the Balasore floods, pre-positioned WASH kits enabled distribution within 36 hours; diarrheal disease alerts dropped in the following week.",
            "faq": [
                {"q": "Do you accept in-kind donations?", "a": "Yes—pre-packed WASH kits and tarps that meet listed specifications."},
                {"q": "How is beneficiary targeting done?", "a": "Through committee-led vulnerability criteria and public lists."}
            ],
            "volunteer": ["Warehouse sorting", "Rapid assessment teams", "Community preparedness drills"],
            "donate": "Emergency fund and kit sponsorship; 80G/12A",
            "financials": "FY 2023–24: INR 5.2 Cr; 88% program, 7% admin, 5% fundraising",
            "contact": "ops@rapidrelief.org | Bhubaneswar"
        },
        {
            "title": "MindWell Initiative",
            "mission": "Expand access to affordable mental health care through community counseling, tele-therapy, and workplace wellbeing.",
            "focus_areas": ["Counseling", "Tele-mental health", "Youth wellbeing", "Suicide prevention"],
            "programs": ["Sliding-scale clinics", "24×7 helpline", "School resilience curriculum", "Gatekeeper training"],
            "geographic_focus": "Delhi NCR, Jaipur, Lucknow; tele-therapy pan-India",
            "beneficiaries": "Low-income adults, adolescents, frontline workers",
            "impact": ["42,000 helpline calls", "11,500 sessions", "350 crisis interventions", "220 institutions trained"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Not registered"],
                "policies": ["Clinical Ethics & Safety", "Informed Consent & Privacy", "Child Safeguarding"]
            },
            "governance": "Clinical oversight committee and anonymized outcomes dashboards",
            "sdgs": ["SDG3 Good Health & Well-being", "SDG8 Decent Work"],
            "monitoring": [
                "Session outcomes (PHQ/GAD) tracked at baseline and follow-ups",
                "Helpline SLAs for response and triage",
                "Institutional trainings feedback loops"
            ],
            "risks": [
                {"risk": "High demand burnout", "mitigation": "Supervision, caseload caps, and rest rosters"},
                {"risk": "Crisis escalation", "mitigation": "Escalation SOPs and hospital referral MoUs"}
            ],
            "partnerships": ["Govt helpline integrators", "Schools & colleges", "ESI/SME workplaces"],
            "kpis": [
                ("PHQ-9 Avg Improvement", "—", "4.1 pts", ">=5 pts"),
                ("Helpline Avg Wait", "90s", "42s", "<=45s"),
                ("Training Satisfaction", "—", "92%", "90%+")
            ],
            "case_study": "A blended model of helpline triage plus four tele-sessions cut absenteeism by 14% in a factory cluster over two quarters.",
            "faq": [
                {"q": "Are sessions confidential?", "a": "Yes—within legal and safety exceptions outlined in consent."},
                {"q": "Do you provide psychiatry?", "a": "We refer to partner psychiatrists when medication is indicated."}
            ],
            "volunteer": ["Peer counselors (trained)", "Helpline support", "School workshops facilitation"],
            "donate": "Sponsor therapy sessions and helpline minutes; 80G/12A",
            "financials": "FY 2023–24: INR 3.1 Cr; 80% program, 12% admin, 8% fundraising",
            "contact": "care@mindwell.org | New Delhi"
        },
        {
            "title": "Sakhi Legal Trust",
            "mission": "Ensure safety and justice for women and girls through legal aid, protection services, and community prevention.",
            "focus_areas": ["Gender-based violence", "Legal aid", "Shelter referrals", "Community prevention"],
            "programs": ["Free legal clinics and court accompaniment", "Protection orders and FIR support", "Paralegal network", "Safe workplace compliance"],
            "geographic_focus": "Maharashtra, Gujarat (urban slums and peri-urban belts)",
            "beneficiaries": "Survivors of domestic violence, workplace harassment, trafficking",
            "impact": ["1,950 clients served", "1,120 protection orders", "85% case resolution", "8,000 citizens sensitized"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Not registered"],
                "policies": ["Survivor-Centered Safeguarding", "Legal Ethics & Conflict of Interest", "Privacy & Evidence Handling"]
            },
            "governance": "Legal ethics oversight; case audits and outcomes tracking",
            "sdgs": ["SDG5 Gender Equality", "SDG16 Peace, Justice & Strong Institutions"],
            "monitoring": [
                "Turnaround time for interim reliefs",
                "Case resolution rates and outcomes",
                "Referral pathway effectiveness (shelters/counselors)"
            ],
            "risks": [
                {"risk": "Retaliation risks", "mitigation": "Safety planning and shelter referrals"},
                {"risk": "Case drop-offs", "mitigation": "Paralegal follow-ups and court accompaniment"}
            ],
            "partnerships": ["DCW/Police cells", "Shelters & counselors", "Legal aid services"],
            "kpis": [
                ("Avg Time to Protection Order", "—", "21 days", "<=14 days"),
                ("Case Resolution Rate", "68%", "85%", "85%+"),
                ("Referral Conversion", "—", "73%", "80%")
            ],
            "case_study": "Combined legal aid and shelter referral secured interim relief in 10 days and stable housing within two months.",
            "faq": [
                {"q": "Are services free?", "a": "Legal clinics are free; court fees handled via case fund when needed."},
                {"q": "Do you handle workplace cases?", "a": "Yes—POSH compliance support and case handling."}
            ],
            "volunteer": ["Pro-bono legal research", "Awareness drives", "Case documentation"],
            "donate": "Case fund and community legal camps; 80G/12A",
            "financials": "FY 2023–24: INR 2.4 Cr; 83% program, 10% admin, 7% fundraising",
            "contact": "support@sakhilegal.org | Mumbai"
        },
        {
            "title": "Rainbow Care Collective",
            "mission": "Advance health, safety, and dignity for LGBTQ+ communities through health services, legal support, and livelihood pathways.",
            "focus_areas": ["Community health", "Mental health", "Legal identity", "Livelihoods"],
            "programs": ["STI screening and PrEP linkages", "Peer counseling", "ID and benefits facilitation", "Micro-enterprise incubation"],
            "geographic_focus": "Bengaluru, Chennai, Hyderabad",
            "beneficiaries": "LGBTQ+ individuals and families, especially youth and trans communities",
            "impact": ["9,200 clinic visits", "1,600 counseling sessions", "1,050 ID issuances", "240 livelihoods incubated"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Not registered"],
                "policies": ["Inclusive HR", "Community Safeguarding", "Data Privacy & Consent"]
            },
            "governance": "Community advisory board and annual program audits",
            "sdgs": ["SDG3 Good Health", "SDG8 Decent Work", "SDG10 Reduced Inequalities"],
            "monitoring": [
                "Clinic utilization and follow-ups",
                "Counseling outcomes (client-reported)",
                "Livelihoods income tracking at 6/12 months"
            ],
            "risks": [
                {"risk": "Stigma limiting access", "mitigation": "Peer navigators and safe spaces"},
                {"risk": "Drop-offs in training", "mitigation": "Stipends and flexible scheduling"}
            ],
            "partnerships": ["Public hospitals", "Legal aid clinics", "Micro-finance partners"],
            "kpis": [
                ("Clinic Follow-up Rate", "—", "72%", "80%"),
                ("ID Issuance Success", "—", "89%", "95%"),
                ("Income Uplift @12m", "—", "26%", "30%")
            ],
            "case_study": "Peer navigation and ID facilitation helped a trans woman secure healthcare and launch a tailoring business within six months.",
            "faq": [
                {"q": "Are services free?", "a": "Clinical services are subsidized; counseling and legal aid free for low-income clients."},
                {"q": "Do you work with employers?", "a": "Yes—sensitization and inclusive hiring pathways."}
            ],
            "volunteer": ["Community outreach", "Mentorship", "Legal camp support"],
            "donate": "Health kits, counseling sponsorships; 80G/12A",
            "financials": "FY 2023–24: INR 2.8 Cr; 84% program, 11% admin, 5% fundraising",
            "contact": "hello@rainbowcare.org | Bengaluru"
        },
        {
            "title": "BlueTap WASH Foundation",
            "mission": "Enable equitable water, sanitation, and hygiene access through rural water systems and school WASH.",
            "focus_areas": ["Safe water", "School sanitation", "Hygiene behavior change", "O&M capacity"],
            "programs": ["Gravity-fed piped schemes", "Community WASH committees", "Girls’ toilets and MHM rooms", "Mason training for upkeep"],
            "geographic_focus": "Jharkhand, Chhattisgarh, Madhya Pradesh",
            "beneficiaries": "Tribal and marginalized villages and government schools",
            "impact": ["75 villages served", "180 school WASH upgrades", "1,200 committee members trained", "28,000 people reached"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Registered"],
                "policies": ["WASH Safeguarding", "Community Contracts & Social Audits", "Data Privacy"]
            },
            "governance": "Community contracts with public dashboards and social audits",
            "sdgs": ["SDG6 Clean Water & Sanitation", "SDG4 Quality Education"],
            "monitoring": [
                "Water quality testing cadence (WHO/IS standards)",
                "School WASH functionality audits",
                "Community committee performance tracking"
            ],
            "risks": [
                {"risk": "O&M breakdowns", "mitigation": "Local mason training and spare-parts funds"},
                {"risk": "Seasonal scarcity", "mitigation": "Source protection and storage"}
            ],
            "partnerships": ["Panchayats", "Jal Jeevan Mission units", "Local CSOs"],
            "kpis": [
                ("Functionality @12m", "—", "88%", "95%"),
                ("Girls’ Toilet Uptime", "—", "92%", "95%"),
                ("Safe Water Compliance", "—", "90%", "95%")
            ],
            "case_study": "Committee-led O&M cut downtime by 60% across 12 villages, sustaining safe water access through peak summer.",
            "faq": [
                {"q": "Do you support small schools?", "a": "Yes—cluster models reduce per-school capex."},
                {"q": "Who owns assets?", "a": "Community/government; the NGO focuses on build and handover."}
            ],
            "volunteer": ["Water testing drives", "Behavior-change campaigns", "School facility audits"],
            "donate": "Sponsor village water points and school WASH; 80G/12A",
            "financials": "FY 2023–24: INR 4.6 Cr; 86% program, 8% admin, 6% fundraising",
            "contact": "info@bluetapwash.org | Ranchi"
        },
        {
            "title": "AgriRoots Livelihoods",
            "mission": "Strengthen smallholder livelihoods via climate-smart agriculture, FPOs, and market linkages.",
            "focus_areas": ["Climate-smart practices", "FPO capacity", "Value chains", "Crop insurance literacy"],
            "programs": ["Demo plots and weather advisories", "FPO governance and credit", "Aggregation and buyers’ meets", "Post-harvest loss reduction"],
            "geographic_focus": "Vidarbha, Marathwada, Bundelkhand",
            "beneficiaries": "Small and marginal farmers, especially women producers",
            "impact": ["22 FPOs strengthened", "14,000 farmers trained", "11% income uplift", "18% loss reduction"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Registered"],
                "policies": ["Farmer Safeguarding & Fair Pricing", "Data Privacy"]
            },
            "governance": "Producer representation on board; open MIS on pricing",
            "sdgs": ["SDG2 Zero Hunger", "SDG8 Decent Work", "SDG13 Climate Action"],
            "monitoring": [
                "Farm income panels (seasonal)",
                "FPO governance scorecards",
                "Post-harvest loss tracking"
            ],
            "risks": [
                {"risk": "Climate shocks", "mitigation": "Crop diversification and advisories"},
                {"risk": "Buyer defaults", "mitigation": "Escrow and graded exposure"}
            ],
            "partnerships": ["Krishi Vigyan Kendras", "Agri-buyers", "Banks/NBFCs"],
            "kpis": [
                ("Avg Income Uplift", "—", "11%", "15%"),
                ("FPO Governance Score", "—", "72/100", "80/100"),
                ("Post-Harvest Loss", "—", "−18%", "−25%")
            ],
            "case_study": "Staggered sowing plus low-cost storage raised net incomes by 19% for 420 farmers in a two-season pilot.",
            "faq": [
                {"q": "Do you finance inputs?", "a": "We facilitate credit via banks and NBFC partners."},
                {"q": "Do you buy produce?", "a": "No—FPOs aggregate and negotiate directly with buyers."}
            ],
            "volunteer": ["Farmer training support", "FPO bookkeeping mentorship", "Market linkage events"],
            "donate": "Farmer training kits and FPO seed grants; 80G/12A",
            "financials": "FY 2023–24: INR 3.9 Cr; 87% program, 9% admin, 4% fundraising",
            "contact": "connect@agriroots.org | Nagpur"
        },
        {
            "title": "Ocean Shield Trust",
            "mission": "Protect coastal ecosystems and strengthen fisher communities through marine conservation and resilient livelihoods.",
            "focus_areas": ["Reef and mangrove restoration", "Sustainable fishing", "Plastic mitigation", "Disaster preparedness"],
            "programs": ["Community reef monitoring", "Mangrove nurseries", "Gear-swap for bycatch reduction", "Ocean plastic collection with buyback"],
            "geographic_focus": "Tamil Nadu and Kerala coasts; island pilots",
            "beneficiaries": "Small-scale fisher families, women SHGs, and youth",
            "impact": ["42 ha mangroves restored", "18 reef transects protected", "420 tons plastic removed", "1,300 households trained"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Registered"],
                "policies": ["Environmental Safeguards", "Community Safeguarding", "Data Transparency"]
            },
            "governance": "Science advisory panel with open-source monitoring data",
            "sdgs": ["SDG14 Life Below Water", "SDG13 Climate Action", "SDG1 No Poverty"],
            "monitoring": [
                "Ecological monitoring (transects, survival rates)",
                "Fish catch quality/effort trends (community logs)",
                "Plastic diversion audits"
            ],
            "risks": [
                {"risk": "Storm damage", "mitigation": "Staggered planting and species mixes"},
                {"risk": "Low community buy-in", "mitigation": "Revenue-sharing from plastic buyback"}
            ],
            "partnerships": ["Fisher cooperatives", "Forest departments", "Marine research labs"],
            "kpis": [
                ("Mangrove Survival @12m", "—", "78%", "85%"),
                ("Bycatch Reduction", "—", "22%", "30%"),
                ("Plastic Diverted (t/yr)", "—", "420", "500")
            ],
            "case_study": "A gear-swap pilot cut bycatch by 27% and improved net margins via selective nets and buyer incentives.",
            "faq": [
                {"q": "Do you pay for plastic?", "a": "Yes—through verified buyback partners at fair rates."},
                {"q": "Who owns restored areas?", "a": "Government/community; stewardship agreements apply."}
            ],
            "volunteer": ["Beach clean-ups", "Nursery care", "Community data logging"],
            "donate": "Adopt-a-reef/mangrove plots; 80G/12A",
            "financials": "FY 2023–24: INR 4.0 Cr; 85% program, 10% admin, 5% fundraising",
            "contact": "team@oceanshield.org | Kochi"
        },
        {
            "title": "SafeStreets Initiative",
            "mission": "Reduce road accidents and fatalities through youth education, enforcement support, and safer street design.",
            "focus_areas": ["Road safety education", "Helmet and seatbelt compliance", "Speed calming", "Black-spot audits"],
            "programs": ["School and college safety clubs", "Enforcement awareness drives", "Junction redesign pilots", "Crash analytics"],
            "geographic_focus": "Delhi NCR, Jaipur, Indore",
            "beneficiaries": "Students, commuters, and delivery riders",
            "impact": ["95 safety clubs", "38% compliance uplift", "21 black spots treated", "12% crash reduction in zones"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Not registered"],
                "policies": ["Volunteer Safety", "Data Privacy", "Child Safeguarding"]
            },
            "governance": "Joint working groups with traffic police; quarterly public results",
            "sdgs": ["SDG3 Good Health", "SDG11 Sustainable Cities"],
            "monitoring": [
                "Before/after compliance studies",
                "Speed and conflict observations",
                "Crash trend analytics with police data"
            ],
            "risks": [
                {"risk": "Non-compliance persistence", "mitigation": "Design + enforcement + education triad"},
                {"risk": "Pilot backlash", "mitigation": "Stakeholder mapping and phased rollouts"}
            ],
            "partnerships": ["Traffic police", "Urban local bodies", "Academic design labs"],
            "kpis": [
                ("Helmet Compliance @Sites", "—", "+38pp", "+45pp"),
                ("85th Percentile Speed", "—", "−12%", "−15%"),
                ("Black-spot Crash Δ", "—", "−12%", "−20%")
            ],
            "case_study": "Post redesign and enforcement, a crash hot-spot saw a 17% drop in injuries within six months.",
            "faq": [
                {"q": "Do you do rural roads?", "a": "We focus on urban corridors with high conflict points."},
                {"q": "Can companies adopt junctions?", "a": "Yes—CSR-supported pilots are welcome."}
            ],
            "volunteer": ["Safety club mentors", "Junction audits", "Awareness events"],
            "donate": "Sponsor safety clubs and junction redesign pilots; 80G/12A",
            "financials": "FY 2023–24: INR 2.2 Cr; 82% program, 12% admin, 6% fundraising",
            "contact": "contact@safestreets.in | Gurugram"
        },
        {
            "title": "HomeBridge Urban Mission",
            "mission": "Support urban homeless individuals with shelter access, identity documentation, and pathways to stable housing.",
            "focus_areas": ["Night shelters", "Identity access", "Social entitlements", "Livelihoods"],
            "programs": ["Outreach vans and navigation to shelters", "ID and bank account facilitation", "Skill linkages", "Transitional rental support"],
            "geographic_focus": "Mumbai, Ahmedabad, Kolkata",
            "beneficiaries": "Homeless adults, migrant workers, elderly without support",
            "impact": ["7,800 shelter navigations", "3,100 IDs issued", "1,250 jobs placed", "420 transitional rentals"],
            "registration": {
                "registrations": ["12A", "80G", "CSR-1", "FCRA: Not registered"],
                "policies": ["Safeguarding for Vulnerable Adults", "Grievance Redressal", "Data Privacy"]
            },
            "governance": "Grievance portal, social returns audits, quarterly board reviews",
            "sdgs": ["SDG1 No Poverty", "SDG11 Sustainable Cities"],
            "monitoring": [
                "Shelter navigation success rates",
                "ID issuance turnaround",
                "Rental stability and job retention at 6/12 months"
            ],
            "risks": [
                {"risk": "Documentation hurdles", "mitigation": "Mobile enrollment camps and case workers"},
                {"risk": "Relapse into homelessness", "mitigation": "Transitional rentals and job retention support"}
            ],
            "partnerships": ["Urban shelter providers", "UIDAI/Bank facilitation centers", "Skilling partners"],
            "kpis": [
                ("ID Issuance TAT (days)", "—", "12", "<=10"),
                ("Rental Stability @6m", "—", "78%", "85%"),
                ("Job Retention @12m", "—", "69%", "75%")
            ],
            "case_study": "A 90‑day plan combining IDs, bank accounts, and placement stabilized 63 clients into rental rooms near job sites.",
            "faq": [
                {"q": "Do you offer shelters?", "a": "We navigate to partner shelters and support documentation."},
                {"q": "What support is available after jobs?", "a": "Follow-ups for retention, finances, and tenancy."}
            ],
            "volunteer": ["ID camps assistance", "Resume workshops", "Shelter outreach"],
            "donate": "Shelter kits and documentation camps; 80G/12A",
            "financials": "FY 2023–24: INR 3.3 Cr; 84% program, 10% admin, 6% fundraising",
            "contact": "hello@homebridge.org | Mumbai"
        },
    ]

    created = []
    for p in profiles:
        created.append(write_pdf(p))
    print("Created PDFs:")
    for p in created:
        print(" -", p)

if __name__ == "__main__":
    main()