<?php

require_once __DIR__ . '/includes/bootstrap.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf($_POST['csrf_token'] ?? null)) {
        set_flash('danger', 'Invalid request token. Please try again.');
        header('Location: /index.php#contact');
        exit;
    }

    $name = trim($_POST['name'] ?? '');
    $email = trim($_POST['email'] ?? '');
    $phone = trim($_POST['phone'] ?? '');
    $message = trim($_POST['message'] ?? '');

    if ($name === '' || !filter_var($email, FILTER_VALIDATE_EMAIL) || $message === '') {
        set_flash('danger', 'Please fill all required fields with valid information.');
    } else {
        $stmt = db()->prepare('INSERT INTO contact_messages (name, email, phone, message) VALUES (?, ?, ?, ?)');
        $stmt->execute([$name, $email, $phone, $message]);
        set_flash('success', 'Thanks! Your message has been sent. We will contact you soon.');
    }

    header('Location: /index.php#contact');
    exit;
}

require __DIR__ . '/includes/header.php';
?>
<section class="hero rounded-3 p-5 mb-4 text-white">
    <h1 class="display-5 fw-bold">Track your project progress in one place</h1>
    <p class="lead mb-4">A beginner-friendly portal where clients can log in, see live project status, and receive updates from your team.</p>
    <a class="btn btn-light btn-lg" href="/signup.php">Get Started</a>
</section>

<section class="mb-5" id="features">
    <h2 class="mb-3">Services & Features</h2>
    <div class="row g-3">
        <div class="col-md-4">
            <div class="card h-100 shadow-sm"><div class="card-body"><h5>Client Contact</h5><p>Collect project requirements quickly through a simple contact form.</p></div></div>
        </div>
        <div class="col-md-4">
            <div class="card h-100 shadow-sm"><div class="card-body"><h5>Secure Client Portal</h5><p>Clients get accounts with hashed passwords and session-based login.</p></div></div>
        </div>
        <div class="col-md-4">
            <div class="card h-100 shadow-sm"><div class="card-body"><h5>Status Tracking</h5><p>View project status, progress percentage, and latest team updates.</p></div></div>
        </div>
    </div>
</section>

<section id="contact" class="mb-4">
    <h2 class="mb-3">Contact Us</h2>
    <form method="post" class="card shadow-sm p-3">
        <input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>">
        <div class="row g-3">
            <div class="col-md-6">
                <label class="form-label">Name *</label>
                <input class="form-control" name="name" required>
            </div>
            <div class="col-md-6">
                <label class="form-label">Email *</label>
                <input class="form-control" type="email" name="email" required>
            </div>
            <div class="col-md-6">
                <label class="form-label">Phone</label>
                <input class="form-control" name="phone">
            </div>
            <div class="col-12">
                <label class="form-label">Message *</label>
                <textarea class="form-control" name="message" rows="4" required></textarea>
            </div>
            <div class="col-12">
                <button class="btn btn-primary" type="submit">Send Message</button>
            </div>
        </div>
    </form>
</section>
<?php require __DIR__ . '/includes/footer.php'; ?>
