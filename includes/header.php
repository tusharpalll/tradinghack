<?php

require_once __DIR__ . '/bootstrap.php';
$flash = get_flash();
$user = current_user();
$appName = config()['app_name'];
?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title><?= e($appName) ?></title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="/assets/css/style.css">
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
        <a class="navbar-brand" href="/index.php"><?= e($appName) ?></a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navMenu">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navMenu">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item"><a class="nav-link" href="/index.php">Home</a></li>
                <?php if (!$user): ?>
                    <li class="nav-item"><a class="nav-link" href="/signup.php">Client Signup</a></li>
                    <li class="nav-item"><a class="nav-link" href="/login.php">Client Login</a></li>
                    <li class="nav-item"><a class="nav-link" href="/admin/login.php">Admin Login</a></li>
                <?php elseif ($user['role'] === 'admin'): ?>
                    <li class="nav-item"><a class="nav-link" href="/admin/dashboard.php">Admin Dashboard</a></li>
                    <li class="nav-item"><a class="nav-link" href="/admin/logout.php">Logout</a></li>
                <?php else: ?>
                    <li class="nav-item"><a class="nav-link" href="/dashboard.php">Dashboard</a></li>
                    <li class="nav-item"><a class="nav-link" href="/logout.php">Logout</a></li>
                <?php endif; ?>
            </ul>
        </div>
    </div>
</nav>

<main class="container py-4">
    <?php if ($flash): ?>
        <div class="alert alert-<?= e($flash['type']) ?>" role="alert">
            <?= e($flash['message']) ?>
        </div>
    <?php endif; ?>
