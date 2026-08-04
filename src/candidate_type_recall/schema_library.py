import re


SCHEMA_LIBRARY = {
    "Business:Declare-Bankruptcy": {
        "definition": "An organization formally declares that it cannot pay its debts or continue operating financially.",
        "trigger_cues": ["bankruptcy", "bankrupt", "insolvency", "filed for bankruptcy"],
        "core_roles": ["Org"]
    },
    "Business:End-Org": {
        "definition": "An organization shuts down, dissolves, or otherwise stops existing.",
        "trigger_cues": ["closed", "shut down", "dissolved", "ceased operations"],
        "core_roles": ["Org", "Place"]
    },
    "Business:Merge-Org": {
        "definition": "Two or more organizations combine into a single organization.",
        "trigger_cues": ["merge", "merged", "acquisition merger", "combine"],
        "core_roles": ["Org"]
    },
    "Business:Start-Org": {
        "definition": "A new organization or company is founded or launched.",
        "trigger_cues": ["founded", "launched", "started", "established"],
        "core_roles": ["Org", "Person", "Place"]
    },
    "Conflict:Attack": {
        "definition": "A violent physical attack, strike, assault, or other hostile act is carried out against a target.",
        "trigger_cues": ["attack", "bombing", "assault", "strike"],
        "core_roles": ["Attacker", "Target", "Victim", "Place", "Instrument"]
    },
    "Conflict:Demonstrate": {
        "definition": "People gather publicly to protest, march, rally, or demonstrate.",
        "trigger_cues": ["protest", "rally", "march", "demonstration"],
        "core_roles": ["Entity", "Place"]
    },
    "Contact:Broadcast": {
        "definition": "Information is communicated publicly through television, radio, online broadcast, or another mass channel.",
        "trigger_cues": ["broadcast", "aired", "televised", "announced publicly"],
        "core_roles": ["Communicator", "Audience"]
    },
    "Contact:Contact": {
        "definition": "People or organizations communicate or make contact in a general way without a more specific communication subtype.",
        "trigger_cues": ["contacted", "spoke", "reached out", "communicated"],
        "core_roles": ["Participant", "Place"]
    },
    "Contact:Correspondence": {
        "definition": "People or organizations communicate by letter, email, message, or other correspondence.",
        "trigger_cues": ["letter", "email", "message", "correspondence"],
        "core_roles": ["Participant", "Place"]
    },
    "Contact:Meet": {
        "definition": "People or organizations meet face to face or hold a meeting.",
        "trigger_cues": ["met", "meeting", "summit", "talks"],
        "core_roles": ["Entity", "Place"]
    },
    "Contact:Phone-Write": {
        "definition": "People communicate through phone calls, writing, letters, email, or similar non-face-to-face communication.",
        "trigger_cues": ["called", "phoned", "wrote", "emailed"],
        "core_roles": ["Entity", "Place"]
    },
    "Justice:Acquit": {
        "definition": "A defendant is formally found not guilty by a legal authority.",
        "trigger_cues": ["acquitted", "found not guilty", "acquittal"],
        "core_roles": ["Defendant", "Adjudicator", "Place"]
    },
    "Justice:Appeal": {
        "definition": "A legal decision or sentence is challenged in a higher court or authority.",
        "trigger_cues": ["appeal", "appealed", "appellate", "challenge ruling"],
        "core_roles": ["Defendant", "Plaintiff", "Adjudicator", "Place"]
    },
    "Justice:Arrest-Jail": {
        "definition": "A person is arrested, detained, taken into custody, or jailed by authorities.",
        "trigger_cues": ["arrested", "detained", "jailed", "taken into custody"],
        "core_roles": ["Person", "Agent", "Place"]
    },
    "Justice:Charge-Indict": {
        "definition": "A person is formally accused, charged, or indicted for a crime.",
        "trigger_cues": ["charged", "indicted", "accused", "indictment"],
        "core_roles": ["Defendant", "Prosecutor", "Adjudicator", "Place"]
    },
    "Justice:Convict": {
        "definition": "A defendant is formally found guilty by a legal authority.",
        "trigger_cues": ["convicted", "found guilty", "conviction"],
        "core_roles": ["Defendant", "Adjudicator", "Place"]
    },
    "Justice:Execute": {
        "definition": "A person is put to death as a legal punishment.",
        "trigger_cues": ["executed", "execution", "put to death"],
        "core_roles": ["Person", "Agent", "Place"]
    },
    "Justice:Extradite": {
        "definition": "A suspect or prisoner is formally transferred to another jurisdiction or country for legal proceedings.",
        "trigger_cues": ["extradited", "extradition", "handed over"],
        "core_roles": ["Person", "Origin", "Destination", "Agent"]
    },
    "Justice:Fine": {
        "definition": "A legal authority imposes a monetary penalty on a person or organization.",
        "trigger_cues": ["fined", "penalty", "ordered to pay", "financial penalty"],
        "core_roles": ["Entity", "Adjudicator", "Place"]
    },
    "Justice:Pardon": {
        "definition": "A legal authority officially forgives a person and cancels or reduces punishment.",
        "trigger_cues": ["pardoned", "pardon", "clemency"],
        "core_roles": ["Defendant", "Adjudicator", "Place"]
    },
    "Justice:Release-Parole": {
        "definition": "A prisoner is released from custody, often on parole or under supervision.",
        "trigger_cues": ["released", "paroled", "freed from prison"],
        "core_roles": ["Person", "Agent", "Place"]
    },
    "Justice:Sentence": {
        "definition": "A court or legal authority formally imposes a punishment on a defendant.",
        "trigger_cues": ["sentenced", "sentence", "jail term", "punishment"],
        "core_roles": ["Defendant", "Adjudicator", "Place"]
    },
    "Justice:Sue": {
        "definition": "A plaintiff files a civil lawsuit against a defendant.",
        "trigger_cues": ["sued", "lawsuit", "suit filed", "litigation"],
        "core_roles": ["Plaintiff", "Defendant", "Adjudicator", "Place"]
    },
    "Justice:Trial-Hearing": {
        "definition": "A trial, hearing, or other formal court proceeding takes place.",
        "trigger_cues": ["trial", "hearing", "court proceedings", "appeared in court"],
        "core_roles": ["Defendant", "Prosecutor", "Adjudicator", "Place"]
    },
    "Life:Be-Born": {
        "definition": "A person is born.",
        "trigger_cues": ["born", "birth", "gave birth"],
        "core_roles": ["Person", "Place"]
    },
    "Life:Die": {
        "definition": "A person dies.",
        "trigger_cues": ["died", "killed", "death", "dead"],
        "core_roles": ["Victim", "Agent", "Place", "Instrument"]
    },
    "Life:Divorce": {
        "definition": "A marriage ends in divorce.",
        "trigger_cues": ["divorced", "divorce", "split up legally"],
        "core_roles": ["Person", "Place"]
    },
    "Life:Injure": {
        "definition": "A person is injured, wounded, or hurt.",
        "trigger_cues": ["injured", "wounded", "hurt", "suffered injuries"],
        "core_roles": ["Victim", "Agent", "Place", "Instrument"]
    },
    "Life:Marry": {
        "definition": "Two people get married or enter a marriage.",
        "trigger_cues": ["married", "wedding", "marriage"],
        "core_roles": ["Person", "Place"]
    },
    "Manufacture:Artifact": {
        "definition": "An artifact, weapon, or other manufactured object is produced or built.",
        "trigger_cues": ["manufactured", "built", "produced", "assembled"],
        "core_roles": ["Agent", "Artifact", "Place"]
    },
    "Movement:Transport": {
        "definition": "A person or artifact is moved or transported from one place to another.",
        "trigger_cues": ["moved", "transported", "sent", "brought"],
        "core_roles": ["Agent", "Artifact", "Person", "Origin", "Destination", "Vehicle"]
    },
    "Movement:Transport-Artifact": {
        "definition": "An object, good, or artifact is transported from one location to another.",
        "trigger_cues": ["shipped", "moved", "transported", "delivered"],
        "core_roles": ["Agent", "Artifact", "Origin", "Destination", "Vehicle"]
    },
    "Movement:Transport-Person": {
        "definition": "A person is transported, moved, or taken from one location to another.",
        "trigger_cues": ["moved", "transported", "taken", "brought"],
        "core_roles": ["Agent", "Person", "Origin", "Destination", "Vehicle"]
    },
    "Personnel:Elect": {
        "definition": "A person is elected to a position or office.",
        "trigger_cues": ["elected", "won election", "voted into office"],
        "core_roles": ["Person", "Entity", "Place"]
    },
    "Personnel:End-Position": {
        "definition": "A person leaves, resigns from, is fired from, or otherwise ends a position.",
        "trigger_cues": ["resigned", "stepped down", "fired", "retired"],
        "core_roles": ["Person", "Entity", "Place"]
    },
    "Personnel:Nominate": {
        "definition": "A person is nominated or proposed for a position or office.",
        "trigger_cues": ["nominated", "nominee", "put forward"],
        "core_roles": ["Person", "Agent", "Entity"]
    },
    "Personnel:Start-Position": {
        "definition": "A person starts, assumes, or is appointed to a position or office.",
        "trigger_cues": ["appointed", "took office", "became", "started as"],
        "core_roles": ["Person", "Entity", "Place"]
    },
    "Transaction:Transaction": {
        "definition": "A general transaction or exchange takes place, covering money, goods, or ownership in a broad sense.",
        "trigger_cues": ["transaction", "deal", "exchange", "sale"],
        "core_roles": ["Giver", "Recipient", "Thing", "Place"]
    },
    "Transaction:Transfer-Money": {
        "definition": "Money is paid, transferred, donated, or otherwise moved from one party to another.",
        "trigger_cues": ["paid", "payment", "donated", "funded"],
        "core_roles": ["Giver", "Recipient", "Beneficiary", "Place"]
    },
    "Transaction:Transfer-Ownership": {
        "definition": "Ownership of an object, property, or organization is transferred from one party to another.",
        "trigger_cues": ["sold", "bought", "purchased", "acquired"],
        "core_roles": ["Giver", "Recipient", "Thing", "Place"]
    }
}


def humanize_type_name(event_type: str) -> str:
    event_type = event_type.replace("-", " ")
    event_type = re.sub(r"(?<!^)(?=[A-Z])", " ", event_type)
    event_type = re.sub(r"\s+", " ", event_type).strip()
    return event_type


def schema_document(entry):
    trigger_text = ", ".join(entry.get("trigger_cues", []))
    role_text = ", ".join(entry.get("core_roles", []))
    parts = [
        f"Event type: {entry['event_type']}",
        f"Definition: {entry['definition']}",
    ]
    if trigger_text:
        parts.append(f"Trigger cues: {trigger_text}")
    if role_text:
        parts.append(f"Core roles: {role_text}")
    return "\n".join(parts)
