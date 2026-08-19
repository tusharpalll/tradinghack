# Client Project Tracker (PHP + MySQL)

A beginner-friendly full-stack web app for agencies/freelancers to track client projects.

## Features

- Public landing page with hero, feature cards, and contact form
- Client signup/login/logout with secure password hashing (`password_hash` / `password_verify`)
- Session-based authentication
- Client dashboard to view assigned projects, status, progress %, and latest update
- Admin login and dashboard to:
  - create client accounts
  - create projects for clients
  - update project status and progress
  - add project updates/notes

## Tech Stack

- PHP (plain PHP, no framework)
- MySQL
- Bootstrap 5 (CDN)

## Folder Structure

- `/index.php` - landing + contact form
- `/signup.php`, `/login.php`, `/logout.php` - client auth
- `/dashboard.php` - client project dashboard
- `/admin/login.php`, `/admin/dashboard.php`, `/admin/logout.php` - admin area
- `/includes` - bootstrap, DB, helpers, layout partials
- `/config/config.php` - app/db config
- `/config/config.example.php` - config template
- `/database/schema.sql` - MySQL schema + default admin seed
- `/assets/css/style.css` - small custom styling

## Local Setup

1. Install PHP 8+ and MySQL.
2. Create database and tables:
   - import `/database/schema.sql` in MySQL (phpMyAdmin or CLI).
3. Update DB settings in `/config/config.php`.
4. Start PHP server from repository root:

   ```bash
   php -S localhost:8000
   ```

5. Open `http://localhost:8000/index.php`.

## Default Admin Login

- Email: `admin@example.com`
- Password: `Admin@123`

> Change this password immediately after first login by editing the user in DB or replacing seed credentials.

## Shared Hosting Deployment (cPanel-friendly)

1. Upload all files to your site root (`public_html` or addon domain root).
2. Create a MySQL database and user from hosting panel.
3. Import `/database/schema.sql`.
4. Update `/config/config.php` with hosting DB credentials.
5. Ensure PHP version is 8.0+.

## Basic Security Notes

- Passwords are hashed.
- SQL queries use prepared statements.
- Session-based auth with session ID regeneration on login.
- CSRF tokens are validated on forms.
- Output is escaped before rendering.
