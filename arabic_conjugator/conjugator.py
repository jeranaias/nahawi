#!/usr/bin/env python3
"""
Arabic Verb Conjugator

Main conjugation engine that handles all root types and verb forms.
Takes a root and produces all conjugated forms.

Handles:
- All 10 verb forms (I-X)
- All root types (sound, hollow, defective, hamzated, doubled, etc.)
- All tenses (past, present, imperative)
- All persons, numbers, genders
- All moods (indicative, subjunctive, jussive)
- Active and passive voice
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .root_types import (
    RootType, RootInfo, classify_root, extract_radicals,
    is_weak, is_waw, is_ya, WEAK_LETTERS
)
from .verb_forms import VerbForm, VERB_FORMS, FormPattern
from .paradigms import (
    Person, Number, Gender, Tense, Mood, Voice,
    PAST_PARADIGM, PRESENT_INDICATIVE, PRESENT_SUBJUNCTIVE,
    PRESENT_JUSSIVE, IMPERATIVE_PARADIGM, ConjugationSlot
)


@dataclass
class ConjugatedForm:
    """A single conjugated verb form."""
    form: str                    # The conjugated word
    root: str                    # The root
    verb_form: VerbForm          # Form I-X
    tense: Tense
    person: Person
    number: Number
    gender: Gender
    mood: Optional[Mood]
    voice: Voice
    label: str                   # e.g., "3ms.past" or "2mp.pres.ind"


class ArabicConjugator:
    """
    Arabic verb conjugator that handles all root types.
    """

    def __init__(self):
        """Initialize the conjugator."""
        pass

    def conjugate_root(
        self,
        root: str,
        verb_form: VerbForm = VerbForm.I,
        tenses: List[Tense] = None,
        include_passive: bool = False
    ) -> List[ConjugatedForm]:
        """
        Conjugate a root in all persons/numbers/genders.

        Args:
            root: The Arabic root (e.g., "كتب", "قول", "بني")
            verb_form: The verb form (I-X)
            tenses: Which tenses to include (default: all)
            include_passive: Include passive voice forms

        Returns:
            List of all conjugated forms
        """
        if tenses is None:
            tenses = [Tense.PAST, Tense.PRESENT, Tense.IMPERATIVE]

        root_info = classify_root(root)
        forms = []

        for tense in tenses:
            if tense == Tense.PAST:
                forms.extend(self._conjugate_past(root_info, verb_form, Voice.ACTIVE))
                if include_passive:
                    forms.extend(self._conjugate_past(root_info, verb_form, Voice.PASSIVE))

            elif tense == Tense.PRESENT:
                for mood in [Mood.INDICATIVE, Mood.SUBJUNCTIVE, Mood.JUSSIVE]:
                    forms.extend(self._conjugate_present(root_info, verb_form, mood, Voice.ACTIVE))
                    if include_passive:
                        forms.extend(self._conjugate_present(root_info, verb_form, mood, Voice.PASSIVE))

            elif tense == Tense.IMPERATIVE:
                forms.extend(self._conjugate_imperative(root_info, verb_form))

        return forms

    def _conjugate_past(
        self,
        root_info: RootInfo,
        verb_form: VerbForm,
        voice: Voice
    ) -> List[ConjugatedForm]:
        """Conjugate past tense for all persons."""
        forms = []
        stem = self._get_past_stem(root_info, verb_form, voice)

        for (person, number, gender), slot in PAST_PARADIGM.items():
            # Apply suffix
            conjugated = self._apply_past_suffix(stem, slot.suffix, root_info)

            forms.append(ConjugatedForm(
                form=conjugated,
                root=root_info.root,
                verb_form=verb_form,
                tense=Tense.PAST,
                person=person,
                number=number,
                gender=gender,
                mood=None,
                voice=voice,
                label=f"{slot.name_en}.past"
            ))

        return forms

    def _conjugate_present(
        self,
        root_info: RootInfo,
        verb_form: VerbForm,
        mood: Mood,
        voice: Voice
    ) -> List[ConjugatedForm]:
        """Conjugate present tense for all persons."""
        forms = []
        stem = self._get_present_stem(root_info, verb_form, voice)

        if mood == Mood.INDICATIVE:
            paradigm = PRESENT_INDICATIVE
        elif mood == Mood.SUBJUNCTIVE:
            paradigm = PRESENT_SUBJUNCTIVE
        else:
            paradigm = PRESENT_JUSSIVE

        for (person, number, gender), slot in paradigm.items():
            # Get the appropriate prefix
            prefix = self._get_present_prefix(verb_form, slot.prefix, voice)

            # Apply prefix and suffix
            conjugated = prefix + stem + slot.suffix

            # Apply morphophonemic rules
            conjugated = self._apply_present_rules(conjugated, root_info, mood, slot)

            mood_str = {Mood.INDICATIVE: "ind", Mood.SUBJUNCTIVE: "subj", Mood.JUSSIVE: "juss"}[mood]

            forms.append(ConjugatedForm(
                form=conjugated,
                root=root_info.root,
                verb_form=verb_form,
                tense=Tense.PRESENT,
                person=person,
                number=number,
                gender=gender,
                mood=mood,
                voice=voice,
                label=f"{slot.name_en}.pres.{mood_str}"
            ))

        return forms

    def _conjugate_imperative(
        self,
        root_info: RootInfo,
        verb_form: VerbForm
    ) -> List[ConjugatedForm]:
        """Conjugate imperative for 2nd person."""
        forms = []
        stem = self._get_imperative_stem(root_info, verb_form)

        for (number, gender), slot in IMPERATIVE_PARADIGM.items():
            # Imperative may need hamzat al-wasl prefix
            prefix = self._get_imperative_prefix(stem, verb_form)

            conjugated = prefix + stem + slot.suffix

            # Apply rules
            conjugated = self._apply_imperative_rules(conjugated, root_info)

            forms.append(ConjugatedForm(
                form=conjugated,
                root=root_info.root,
                verb_form=verb_form,
                tense=Tense.IMPERATIVE,
                person=Person.SECOND,
                number=number,
                gender=gender,
                mood=None,
                voice=Voice.ACTIVE,
                label=f"{slot.name_en}.imp"
            ))

        return forms

    # ============================================
    # STEM GENERATION BY ROOT TYPE
    # ============================================

    def _get_past_stem(
        self,
        root_info: RootInfo,
        verb_form: VerbForm,
        voice: Voice
    ) -> str:
        """
        Get the past tense stem for a root.
        This is where root type specific rules apply.
        """
        letters = root_info.letters
        rt = root_info.root_type

        if verb_form == VerbForm.I:
            return self._form_I_past_stem(root_info, voice)
        elif verb_form == VerbForm.II:
            return self._form_II_past_stem(root_info, voice)
        elif verb_form == VerbForm.III:
            return self._form_III_past_stem(root_info, voice)
        elif verb_form == VerbForm.IV:
            return self._form_IV_past_stem(root_info, voice)
        elif verb_form == VerbForm.V:
            return self._form_V_past_stem(root_info, voice)
        elif verb_form == VerbForm.VI:
            return self._form_VI_past_stem(root_info, voice)
        elif verb_form == VerbForm.VII:
            return self._form_VII_past_stem(root_info, voice)
        elif verb_form == VerbForm.VIII:
            return self._form_VIII_past_stem(root_info, voice)
        elif verb_form == VerbForm.X:
            return self._form_X_past_stem(root_info, voice)
        else:
            # Form IX is rare, use Form I pattern
            return self._form_I_past_stem(root_info, voice)

    def _form_I_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form I past stem: فَعَلَ"""
        letters = root_info.letters
        rt = root_info.root_type

        if rt == RootType.SOUND or rt in [RootType.HAMZATED_FIRST, RootType.HAMZATED_MIDDLE, RootType.HAMZATED_LAST]:
            # Sound: فَعَلَ
            if voice == Voice.ACTIVE:
                return f"{letters[0]}{letters[1]}{letters[2]}"
            else:  # Passive: فُعِلَ
                return f"{letters[0]}{letters[1]}{letters[2]}"

        elif rt == RootType.DOUBLED:
            # Doubled: فَعَّ (شَدَّ)
            if voice == Voice.ACTIVE:
                return f"{letters[0]}{letters[1]}"  # Will add shadda
            else:
                return f"{letters[0]}{letters[1]}"

        elif rt in [RootType.HOLLOW_WAW, RootType.HOLLOW_YA]:
            # Hollow: فَالَ (قَالَ، بَاعَ)
            if voice == Voice.ACTIVE:
                if rt == RootType.HOLLOW_WAW:
                    return f"{letters[0]}ا{letters[2]}"  # قال
                else:
                    return f"{letters[0]}ا{letters[2]}"  # باع
            else:
                # Passive: قِيلَ، بِيعَ
                return f"{letters[0]}ي{letters[2]}"

        elif rt in [RootType.DEFECTIVE_WAW, RootType.DEFECTIVE_YA]:
            # Defective: ends in ى or ا
            if voice == Voice.ACTIVE:
                if rt == RootType.DEFECTIVE_WAW:
                    return f"{letters[0]}{letters[1]}ا"  # دَعَا
                else:
                    return f"{letters[0]}{letters[1]}ى"  # رَمَى
            else:
                return f"{letters[0]}{letters[1]}ي"

        elif rt in [RootType.ASSIMILATED_WAW, RootType.ASSIMILATED_YA]:
            # Assimilated: first letter و or ي
            # In Form I past, usually regular: وَجَدَ
            return f"{letters[0]}{letters[1]}{letters[2]}"

        elif rt == RootType.DOUBLE_WEAK_SEPARATED:
            # لفيف مفروق: وَقَى
            return f"{letters[0]}{letters[1]}ى"

        elif rt == RootType.DOUBLE_WEAK_ADJACENT:
            # لفيف مقرون: رَوَى
            return f"{letters[0]}{letters[1]}ى"

        elif rt == RootType.QUADRILITERAL:
            # Quadriliteral: فَعْلَلَ (دَحْرَجَ)
            return f"{letters[0]}{letters[1]}{letters[2]}{letters[3]}"

        else:
            # Default
            return ''.join(letters)

    def _form_II_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form II past stem: فَعَّلَ (doubled middle)"""
        letters = root_info.letters
        # Form II doubles the middle radical
        return f"{letters[0]}{letters[1]}{letters[1]}{letters[2]}"

    def _form_III_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form III past stem: فَاعَلَ (long a after first)"""
        letters = root_info.letters
        return f"{letters[0]}ا{letters[1]}{letters[2]}"

    def _form_IV_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form IV past stem: أَفْعَلَ (hamza prefix)"""
        letters = root_info.letters
        if voice == Voice.ACTIVE:
            return f"أ{letters[0]}{letters[1]}{letters[2]}"
        else:  # Passive: أُفْعِلَ
            return f"أ{letters[0]}{letters[1]}{letters[2]}"

    def _form_V_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form V past stem: تَفَعَّلَ (ta prefix + doubled middle)"""
        letters = root_info.letters
        return f"ت{letters[0]}{letters[1]}{letters[1]}{letters[2]}"

    def _form_VI_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form VI past stem: تَفَاعَلَ (ta prefix + long a)"""
        letters = root_info.letters
        return f"ت{letters[0]}ا{letters[1]}{letters[2]}"

    def _form_VII_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form VII past stem: اِنْفَعَلَ (in prefix)"""
        letters = root_info.letters
        return f"ان{letters[0]}{letters[1]}{letters[2]}"

    def _form_VIII_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form VIII past stem: اِفْتَعَلَ (i + ta infix)"""
        letters = root_info.letters
        return f"ا{letters[0]}ت{letters[1]}{letters[2]}"

    def _form_X_past_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form X past stem: اِسْتَفْعَلَ (ista prefix)"""
        letters = root_info.letters
        return f"است{letters[0]}{letters[1]}{letters[2]}"

    # ============================================
    # PRESENT TENSE STEMS
    # ============================================

    def _get_present_stem(
        self,
        root_info: RootInfo,
        verb_form: VerbForm,
        voice: Voice
    ) -> str:
        """Get present tense stem (without prefix)."""
        letters = root_info.letters
        rt = root_info.root_type

        if verb_form == VerbForm.I:
            return self._form_I_present_stem(root_info, voice)
        elif verb_form == VerbForm.II:
            # يُفَعِّلُ
            return f"{letters[0]}{letters[1]}{letters[1]}{letters[2]}"
        elif verb_form == VerbForm.III:
            # يُفَاعِلُ
            return f"{letters[0]}ا{letters[1]}{letters[2]}"
        elif verb_form == VerbForm.IV:
            # يُفْعِلُ
            return f"{letters[0]}{letters[1]}{letters[2]}"
        elif verb_form == VerbForm.V:
            # يَتَفَعَّلُ
            return f"ت{letters[0]}{letters[1]}{letters[1]}{letters[2]}"
        elif verb_form == VerbForm.VI:
            # يَتَفَاعَلُ
            return f"ت{letters[0]}ا{letters[1]}{letters[2]}"
        elif verb_form == VerbForm.VII:
            # يَنْفَعِلُ
            return f"ن{letters[0]}{letters[1]}{letters[2]}"
        elif verb_form == VerbForm.VIII:
            # يَفْتَعِلُ
            return f"{letters[0]}ت{letters[1]}{letters[2]}"
        elif verb_form == VerbForm.X:
            # يَسْتَفْعِلُ
            return f"ست{letters[0]}{letters[1]}{letters[2]}"
        else:
            return ''.join(letters)

    def _form_I_present_stem(self, root_info: RootInfo, voice: Voice) -> str:
        """Form I present stem: ـفْعَلُ or ـفْعِلُ or ـفْعُلُ"""
        letters = root_info.letters
        rt = root_info.root_type

        if rt in [RootType.HOLLOW_WAW, RootType.HOLLOW_YA]:
            # Hollow: يَفُولُ، يَبِيعُ → stem without middle vowel in some forms
            if rt == RootType.HOLLOW_WAW:
                return f"{letters[0]}و{letters[2]}"  # يقول
            else:
                return f"{letters[0]}ي{letters[2]}"  # يبيع

        elif rt in [RootType.DEFECTIVE_WAW, RootType.DEFECTIVE_YA]:
            # Defective: يَدْعُو، يَرْمِي
            return f"{letters[0]}{letters[1]}"  # Final letter changes based on suffix

        elif rt in [RootType.ASSIMILATED_WAW]:
            # Assimilated waw: يَجِدُ (و drops in present)
            return f"{letters[1]}{letters[2]}"

        else:
            # Regular: يَفْعَلُ
            return f"{letters[0]}{letters[1]}{letters[2]}"

    def _get_present_prefix(self, verb_form: VerbForm, base_prefix: str, voice: Voice) -> str:
        """Get the present tense prefix, adjusted for verb form."""
        # Forms II, III, IV use damma on prefix: يُفَعِّلُ، يُفَاعِلُ، يُفْعِلُ
        if verb_form in [VerbForm.II, VerbForm.III, VerbForm.IV]:
            return base_prefix  # The vowel is implicit
        else:
            return base_prefix

    def _get_imperative_stem(self, root_info: RootInfo, verb_form: VerbForm) -> str:
        """Get imperative stem."""
        # Imperative is based on jussive minus prefix
        return self._get_present_stem(root_info, verb_form, Voice.ACTIVE)

    def _get_imperative_prefix(self, stem: str, verb_form: VerbForm) -> str:
        """Get imperative prefix (hamzat al-wasl if needed)."""
        # Form I: اِفْعَلْ (hamzat al-wasl)
        # Forms II, III, IV: no hamza needed
        if verb_form == VerbForm.I:
            return "ا"
        else:
            return ""

    # ============================================
    # MORPHOPHONEMIC RULES
    # ============================================

    def _apply_past_suffix(self, stem: str, suffix: str, root_info: RootInfo) -> str:
        """Apply past tense suffix with morphophonemic adjustments."""
        rt = root_info.root_type

        # Defective verbs: final letter changes before consonant suffixes
        if rt in [RootType.DEFECTIVE_WAW, RootType.DEFECTIVE_YA]:
            if suffix and suffix[0] in 'تن':  # Consonant-initial suffix
                # رَمَى + تُ = رَمَيْتُ
                stem = stem[:-1] + 'ي'

        # Doubled verbs: may need to separate doubled letters
        if rt == RootType.DOUBLED:
            if suffix and suffix[0] in 'تن':
                # شَدَّ + تُ = شَدَدْتُ
                stem = stem + stem[-1]

        return stem + suffix

    def _apply_present_rules(
        self,
        form: str,
        root_info: RootInfo,
        mood: Mood,
        slot: ConjugationSlot
    ) -> str:
        """Apply morphophonemic rules to present tense."""
        rt = root_info.root_type

        # Defective verbs: final letter depends on suffix
        if rt in [RootType.DEFECTIVE_WAW, RootType.DEFECTIVE_YA]:
            # يَرْمِي, يَدْعُو, يَرْمُون, يَدْعُون
            if slot.suffix == 'ون':
                form = form[:-2] + 'ون'  # Replace last part
            elif slot.suffix == 'وا':
                form = form[:-2] + 'وا'
            elif not slot.suffix:  # Singular forms
                if rt == RootType.DEFECTIVE_YA:
                    if mood == Mood.JUSSIVE:
                        form = form[:-1]  # يَرْمِ
                    else:
                        form = form + 'ي'  # يَرْمِي
                else:
                    form = form + 'و'  # يَدْعُو

        return form

    def _apply_imperative_rules(self, form: str, root_info: RootInfo) -> str:
        """Apply morphophonemic rules to imperative."""
        return form

    # ============================================
    # HIGH-LEVEL INTERFACE
    # ============================================

    def get_all_forms(self, root: str, verb_form: VerbForm = VerbForm.I) -> Dict[str, str]:
        """
        Get a dictionary of all conjugated forms for a root.

        Returns:
            Dict mapping labels to forms, e.g., {"3ms.past": "كَتَبَ", ...}
        """
        forms = self.conjugate_root(root, verb_form)
        return {f.label: f.form for f in forms}

    def get_past_feminine(self, root: str, verb_form: VerbForm = VerbForm.I) -> Tuple[str, str]:
        """
        Get 3ms and 3fs past tense forms (for verb feminization errors).

        Returns:
            (masculine_form, feminine_form)
        """
        forms = self.conjugate_root(root, verb_form, tenses=[Tense.PAST])
        masc = next((f.form for f in forms if f.label == "3ms.past"), None)
        fem = next((f.form for f in forms if f.label == "3fs.past"), None)
        return (masc, fem)

    def get_present_plural(self, root: str, verb_form: VerbForm = VerbForm.I) -> Tuple[str, str]:
        """
        Get indicative and subjunctive 3mp forms (for ون vs وا errors).

        Returns:
            (indicative_form, subjunctive_form)
        """
        indic_forms = self._conjugate_present(
            classify_root(root), verb_form, Mood.INDICATIVE, Voice.ACTIVE
        )
        subj_forms = self._conjugate_present(
            classify_root(root), verb_form, Mood.SUBJUNCTIVE, Voice.ACTIVE
        )

        indic = next((f.form for f in indic_forms if f.label == "3mp.pres.ind"), None)
        subj = next((f.form for f in subj_forms if f.label == "3mp.pres.subj"), None)

        return (indic, subj)


# ============================================
# TESTING
# ============================================

if __name__ == '__main__':
    print("="*70)
    print("ARABIC CONJUGATOR TEST")
    print("="*70)

    conjugator = ArabicConjugator()

    test_roots = [
        ("كتب", "Sound", "to write"),
        ("قول", "Hollow-Waw", "to say"),
        ("بيع", "Hollow-Ya", "to sell"),
        ("رمي", "Defective-Ya", "to throw"),
        ("دعو", "Defective-Waw", "to call"),
        ("وجد", "Assimilated", "to find"),
        ("أخذ", "Hamzated", "to take"),
        ("شدد", "Doubled", "to tighten"),
    ]

    for root, root_type, meaning in test_roots:
        print(f"\n{'='*50}")
        print(f"Root: {root} ({root_type}) - {meaning}")
        print("="*50)

        # Get past masculine/feminine
        masc, fem = conjugator.get_past_feminine(root)
        print(f"Past: {masc} (he) / {fem} (she)")

        # Get present indicative/subjunctive
        indic, subj = conjugator.get_present_plural(root)
        print(f"Present 3mp: {indic} (ind) / {subj} (subj)")

        # Show all past forms
        print("\nAll past forms:")
        forms = conjugator.conjugate_root(root, VerbForm.I, tenses=[Tense.PAST])
        for f in forms:
            print(f"  {f.label:12} {f.form}")
