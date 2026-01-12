"""
Arabic Verb Conjugator

Conjugates Arabic verbs from their roots across all persons and tenses.
Supports Form I (basic trilateral) and handles common irregular verbs.
"""

from typing import Dict, Optional
from .data import (
    ROOTS, IRREGULAR_VERBS,
    FORM_I_PAST, FORM_I_PRESENT_PREFIX, FORM_I_PRESENT_SUFFIX
)


class VerbConjugator:
    """
    Conjugates Arabic verbs from roots.

    Usage:
        conj = VerbConjugator()
        verb = conj.conjugate('كتب', 'past', '3fs')  # كتبت
        verb = conj.conjugate('ذهب', 'present', '3mp')  # يذهبون
    """

    def __init__(self):
        self.roots = ROOTS
        self.irregular = IRREGULAR_VERBS

    def conjugate(self, root: str, tense: str = 'past', person: str = '3ms',
                  form: int = 1) -> Optional[str]:
        """
        Conjugate a verb from its root.

        Args:
            root: The 3-letter root (e.g., 'كتب')
            tense: 'past', 'present', or 'imperative'
            person: '1s', '1p', '2ms', '2fs', '2mp', '2fp', '3ms', '3fs', '3mp', '3fp', etc.
            form: Verb form (1-10), default is Form I

        Returns:
            The conjugated verb form
        """
        # Check irregular verbs first
        if root in self.irregular:
            if tense in self.irregular[root]:
                if person in self.irregular[root][tense]:
                    return self.irregular[root][tense][person]

        # For now, only support Form I
        if form != 1:
            return None

        # Get root type
        root_type = self.roots.get(root, {}).get('type', 'sound')

        if tense == 'past':
            return self._conjugate_past(root, person, root_type)
        elif tense == 'present':
            return self._conjugate_present(root, person, root_type)
        elif tense == 'imperative':
            return self._conjugate_imperative(root, person, root_type)

        return None

    def _conjugate_past(self, root: str, person: str, root_type: str) -> str:
        """Conjugate past tense (Form I)."""
        if len(root) != 3:
            return root

        r1, r2, r3 = root[0], root[1], root[2]

        # Handle different root types
        if root_type == 'sound':
            base = root
            suffix = FORM_I_PAST.get(person, '')

            # Adjust for person
            if person in ('3ms',):
                return base
            elif person in ('3fs', '2ms', '2fs', '1s'):
                return base + 'ت'
            elif person == '3mp':
                return base + 'وا'
            elif person == '3fp':
                return base + 'ن'
            elif person == '2mp':
                return base + 'تم'
            elif person == '2fp':
                return base + 'تن'
            elif person == '1p':
                return base + 'نا'
            elif person == '3md':
                return base + 'ا'
            elif person == '3fd':
                return base + 'تا'
            elif person == '2md':
                return base + 'تما'

        elif root_type == 'hollow':
            # Hollow verbs (middle radical is و or ي): قال، نام، زار
            if person in ('3ms',):
                return root  # قال
            elif person in ('3fs', '2ms', '2fs', '1s'):
                # قُلْت (middle radical disappears)
                return r1 + r3 + 'ت'
            elif person == '3mp':
                return root + 'وا'  # قالوا
            elif person == '3fp':
                return r1 + r3 + 'ن'  # قُلن
            elif person == '1p':
                return r1 + r3 + 'نا'  # قُلنا
            elif person == '2mp':
                return r1 + r3 + 'تم'

        elif root_type == 'defective':
            # Defective verbs (final radical is و or ي): مشى، رمى
            if person in ('3ms',):
                return root  # مشى
            elif person == '3fs':
                return r1 + r2 + 'ت'  # مشت
            elif person in ('2ms', '2fs', '1s'):
                return r1 + r2 + 'يت'  # مشيت
            elif person == '3mp':
                return r1 + r2 + 'وا'  # مشوا
            elif person == '3fp':
                return r1 + r2 + 'ين'  # مشين
            elif person == '1p':
                return r1 + r2 + 'ينا'  # مشينا

        return root + FORM_I_PAST.get(person, '')

    def _conjugate_present(self, root: str, person: str, root_type: str) -> str:
        """Conjugate present tense (Form I)."""
        if len(root) != 3:
            return root

        r1, r2, r3 = root[0], root[1], root[2]
        prefix = FORM_I_PRESENT_PREFIX.get(person, 'ي')
        suffix = FORM_I_PRESENT_SUFFIX.get(person, '')

        if root_type == 'sound':
            # Standard: يَفْعَلُ pattern
            stem = r1 + r2 + r3
            return prefix + stem + suffix

        elif root_type == 'hollow':
            # Hollow: يقول، يقوم، ينام
            # The middle radical appears as و or ي
            # For simplicity, use و
            stem = r1 + 'و' + r3
            return prefix + stem + suffix

        elif root_type == 'defective':
            # Defective: يمشي، يرمي
            stem = r1 + r2 + 'ي'
            if suffix in ('ون', 'ين'):
                stem = r1 + r2  # يمشون (not يمشيون)
            return prefix + stem + suffix

        elif root_type == 'assimilated':
            # Assimilated (first radical is و): يجد، يصل
            # The و disappears in present
            stem = r2 + r3
            return prefix + stem + suffix

        return prefix + root + suffix

    def _conjugate_imperative(self, root: str, person: str, root_type: str) -> str:
        """Conjugate imperative (Form I)."""
        if len(root) != 3:
            return root

        r1, r2, r3 = root[0], root[1], root[2]

        # Imperative only exists for 2nd person
        if not person.startswith('2'):
            return None

        if root_type == 'sound':
            base = 'ا' + r1 + r2 + r3
            if person == '2ms':
                return base
            elif person == '2fs':
                return base + 'ي'
            elif person == '2mp':
                return base + 'وا'
            elif person == '2fp':
                return base + 'ن'

        return 'ا' + root

    def get_all_forms(self, root: str, form: int = 1) -> Dict[str, Dict[str, str]]:
        """Get all conjugated forms of a verb."""
        persons = ['1s', '1p', '2ms', '2fs', '2mp', '3ms', '3fs', '3mp', '3fp']
        tenses = ['past', 'present']

        result = {tense: {} for tense in tenses}

        for tense in tenses:
            for person in persons:
                conj = self.conjugate(root, tense, person, form)
                if conj:
                    result[tense][person] = conj

        return result

    def get_feminine_form(self, verb: str, tense: str = None) -> Optional[str]:
        """Get the feminine form of a verb."""
        # Check irregular verbs
        for root, forms in self.irregular.items():
            for t, persons in forms.items():
                if tense and t != tense:
                    continue
                for person, form in persons.items():
                    if verb == form and 'm' in person:
                        fem_person = person.replace('m', 'f')
                        if fem_person in persons:
                            return persons[fem_person]

        # Heuristic approach
        if not tense:
            # Detect tense
            if verb[0] in 'يتنأ':
                tense = 'present'
            else:
                tense = 'past'

        if tense == 'past':
            if not verb.endswith('ت') and not verb.endswith('وا') and not verb.endswith('ن'):
                return verb + 'ت'
        elif tense == 'present':
            if verb.startswith('ي'):
                return 'ت' + verb[1:]

        return verb

    def get_plural_form(self, verb: str, gender: str = 'masc', tense: str = None) -> Optional[str]:
        """Get the plural form of a verb."""
        if not tense:
            if verb[0] in 'يتنأ':
                tense = 'present'
            else:
                tense = 'past'

        if tense == 'past':
            if verb.endswith('ت'):
                if gender == 'masc':
                    return verb[:-1] + 'وا'
                else:
                    return verb[:-1] + 'ن'
            elif not verb.endswith('وا') and not verb.endswith('ن'):
                if gender == 'masc':
                    return verb + 'وا'
                else:
                    return verb + 'ن'
        elif tense == 'present':
            if not verb.endswith('ون') and not verb.endswith('ن'):
                if gender == 'masc':
                    return verb + 'ون'
                else:
                    return verb + 'ن'

        return verb
