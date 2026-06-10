import getpass
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

User = get_user_model()

ACCOUNTS = [
    {'username': 'PU', 'is_staff': False, 'label': 'regular user'},
    {'username': 'GINA', 'is_staff': True, 'label': 'admin'},
]


class Command(BaseCommand):
    help = 'Create the PU (regular) and GINA (admin) user accounts'

    def handle(self, *args, **options):
        for acct in ACCOUNTS:
            username = acct['username']
            if User.objects.filter(username__iexact=username).exists():
                self.stdout.write(self.style.WARNING(
                    f"  Account '{username}' already exists — skipping."
                ))
                continue

            self.stdout.write(f"\nCreating {acct['label']} account: {username}")
            while True:
                password = getpass.getpass(f"  Enter password for {username}: ")
                confirm = getpass.getpass(f"  Confirm password for {username}: ")
                if password == confirm:
                    break
                self.stdout.write(self.style.ERROR("  Passwords don't match. Try again."))

            user = User.objects.create_user(
                username=username,
                password=password,
                is_staff=acct['is_staff'],
            )
            role = 'admin' if acct['is_staff'] else 'regular'
            self.stdout.write(self.style.SUCCESS(
                f"  Created {role} account '{user.username}' (id={user.pk})"
            ))

        self.stdout.write(self.style.SUCCESS('\nDone.'))
